#include "CPUHashSTL.h"


CPUHashSTL::CPUHashSTL(LYMesh* m, int numBuckets) :
originalMesh(m),
nBuckets(numBuckets),
influenceRadius(0.02f),
cellSize(0.02f)
{
	originalPoints.resize(m->getNumVertices());
	originalPoints = *m->getVertices();
	m_touched = false;

	for (auto it = originalPoints.begin(); it != originalPoints.end(); ++it)
	{
		lookup.insert(std::make_pair(hashFunction(getGridPos(it->m_pos)), *it));
	}
}


CPUHashSTL::~CPUHashSTL(void)
{
}

void CPUHashSTL::update()
{

	if (m_dirtyPos)
	{
		std::vector<LYVertex> v;
		v.reserve(lookup.size());

		if (m_updatePositions) {
			for (LYVertexLookup::const_iterator& elem = lookup.begin(); elem != lookup.end(); ++elem)
				v.push_back(elem->second);

			originalMesh->setPositions(v);
			this->resetPositions();
		}
	}
}

void CPUHashSTL::clear()
{

}

void CPUHashSTL::dump()
{

	std::vector<LYVertex> v;
	v.reserve(lookup.size());

	for (LYVertexLookup::const_iterator& elem = lookup.begin(); elem != lookup.end(); ++elem)
		v.push_back(elem->second);
}

float3 CPUHashSTL::calculateFeedbackUpdateProxy(Collider *pos)
{
	float3 colliderPos = pos->hapticPosition;
	float3 Pseed = make_float3(0.0f);
	float3 dP = make_float3(0.0f);
	float3 Ax, Nx;
	float Fx;

	if (!m_touched){
		calculateCollisions(colliderPos);
		Ax = m_Ax/m_w_tot;
		float nLength = length(m_Nx);
		Nx = m_Nx/nLength;

		Fx = dot(Nx, colliderPos - Ax);

		if (Fx < 0.0f){
			m_touched = true;
			Pseed = Ax;
			pos->scpPosition = Pseed;
			pos->surfaceTgPlane = Nx;
			m_forceFeedback = make_float4((Pseed - colliderPos), 0.0f);
			if (length(m_forceFeedback) > 0.03f) m_dirtyPos = true;
			return make_float3(m_forceFeedback);
		} else {
			pos->scpPosition = colliderPos;
			m_touched = false;
		}
	} else {
		float dist = dot(colliderPos - pos->scpPosition, pos->surfaceTgPlane);
		if (dist <= -0.000001f)
		{
			Pseed = colliderPos - dist*pos->surfaceTgPlane;
			int err_iterations(0);
			int iterations(0);
			do{
				while(calculateCollisions(Pseed) < 0.001f && ++err_iterations < 4);
				Ax = m_Ax/m_w_tot;
				Nx = m_Nx;
				float dNx = length(Nx);
				Nx /= dNx;
				Fx = dot(Nx, Pseed - Ax);
				dP.x = -Fx * Nx.x;
				dP.y = -Fx * Nx.y;
				dP.z = -Fx * Nx.z;
				dP = dP/fmaxf(dNx, 0.1f);
				dP *= 0.001f;
				Pseed += dP;
			} while (length(dP) > 0.001f && ++iterations < 4);
			pos->scpPosition = Pseed;
			pos->surfaceTgPlane = Nx;
			m_forceFeedback = make_float4((Pseed - colliderPos), 0.0f);
			if (length(m_forceFeedback) > 0.03f) m_dirtyPos = true;

			return make_float3(m_forceFeedback);
		} else {
			printf("Contact lost!  ");
			Ax = pos->scpPosition;
			Nx = pos->surfaceTgPlane;
			Fx = dist;
			printf("Ax = (%f, %f, %f)  ", Ax.x, Ax.y, Ax.z);
			printf("Nx = (%f, %f, %f)  ", Nx.x, Nx.y, Nx.z);
			printf("Fx = %f\n", Fx);
			m_touched = false;
			pos->scpPosition = colliderPos;
		}
	}

	m_forceFeedback = make_float4(0.0f);
	return make_float3(m_forceFeedback);
}

const float CPUHashSTL::wendlandWeight(const float& dist) const
{
	float a = 1-dist;
	return ( (a*a*a*a) * ((4*a) + 1) );
}

float CPUHashSTL::calculateCollisions(const float3 pos)
{
	m_Nx = make_float3(0.0f);
	m_Ax = make_float3(0.0f);
	m_w_tot = 0.0f;

	uint3 gridPos = getGridPos(pos);
	int nSize = (int) glm::ceil( influenceRadius / cellSize );

	//printf("gridPos = %d %d %d\n", gridPos.x, gridPos.y, gridPos.z);
	//printf("nSize = %f %f %f\n", nSize.x, nSize.y, nSize.z);	

	// For all the voxels around currentVoxel go to the neighbors and launch threads on each vertex on them.

	std::vector<LYVertex> neighbors;

	for(int z=-nSize; z<=nSize; z++) {
		for(int y=-nSize; y<=nSize; y++) {
			for(int x=-nSize; x<=nSize; x++) {
				uint3 gridNeigh = gridPos + make_uint3(x,y,z);
				auto bucket = lookup.equal_range(hashFunction(gridNeigh));
				for (LYVertexLookup::iterator it = bucket.first; it != bucket.second; ++it){
					float3 npos;
					float w = 0.0f;
					npos = it->second.m_pos - pos;
					float dist = length(npos);
					float R = influenceRadius;

					if (dist < R) {
						w = wendlandWeight(dist/R);
						m_Ax += w * it->second.m_pos;
						m_Nx += w * it->second.m_normal;
						m_w_tot += w;
					}
				}
			}
		}
	}
	
	return m_w_tot;
}

void CPUHashSTL::setInfluenceRadius(float r)
{
	influenceRadius = r;
}

void CPUHashSTL::toggleUpdatePositions()
{
	m_dirtyPos = !m_dirtyPos;
}

void CPUHashSTL::resetPositions()
{
	lookup.clear();
	for (auto it = originalPoints.begin(); it != originalPoints.end(); ++it)
	{
		lookup.insert(std::make_pair(hashFunction(getGridPos(it->m_pos)), *it));
	}
}

const float3 CPUHashSTL::translateRange(const float3& value, const float3& lMin, const float3& lMax, const float3 &rMin, const float3 &rMax) const
{
	float3 lSpan = lMax-lMin;
	float3 rSpan = rMax-rMin;

	float3 valueScaled = (value - lMin) / (lSpan);

	return rMin + (valueScaled * rSpan); 
}

const uint3 CPUHashSTL::getGridPos(const float3& p) const{
	uint3 r;
	const float3 minP = make_float3(originalMesh->getMinPoint().x, originalMesh->getMinPoint().y, originalMesh->getMinPoint().z);
	const float3 maxP = make_float3(originalMesh->getMaxPoint().x, originalMesh->getMaxPoint().y, originalMesh->getMaxPoint().z);
	float3 v = translateRange(p, minP, maxP, make_float3(0.0f), make_float3(1.0f));
	r.x= (uint) floor(v.x/cellSize);
	r.y= (uint) floor(v.y/cellSize);
	r.z= (uint) floor(v.z/cellSize);
	return r;
}

const uint CPUHashSTL::hashFunction(const uint3& p) const
{
	const uint p1 = 73856093;
	const uint p2 = 19349663;
	const uint p3 = 83492791;

	return (p.x*p1 ^ p.y*p2 ^ p.z*p3)%nBuckets;
}
