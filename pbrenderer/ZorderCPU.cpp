#include "ZorderCPU.h"


ZorderCPU::ZorderCPU(LYMesh *mesh)
{
	sortedPoints.resize(mesh->getNumVertices());
	sortedPoints = *mesh->getVertices();
	zIndeces.resize(sortedPoints.size());
	updateStructure();
}


ZorderCPU::~ZorderCPU(void)
{
}

bool ZorderCPU::comparePoints(const LYVertex& p1, const LYVertex& p2)
{
	return (p1.m_pos.x < p2.m_pos.x &&
			p1.m_pos.y < p2.m_pos.y &&
			p1.m_pos.z < p2.m_pos.z );
}

void ZorderCPU::updateStructure()
{

	auto minValue = std::min_element(sortedPoints.begin(), sortedPoints.end());

	auto maxValue = std::max_element(sortedPoints.begin(), sortedPoints.end());
	{
		class zOrderCmp
		{
		public:
			// The required constructor for this example.
			explicit zOrderCmp(float3 mini, float3 Max)
			{
				m_dimension = Max - mini;
			}

			// 
			bool operator()(const LYVertex &p1, const LYVertex &p2) const
			{
				float3 remapPos = p1.m_pos / m_dimension;
				uint tmp;
				float x = std::min(std::max(remapPos.x * 1024.0f, 0.0f), 1023.0f);
				float y = std::min(std::max(remapPos.y * 1024.0f, 0.0f), 1023.0f);
				float z = std::min(std::max(remapPos.z * 1024.0f, 0.0f), 1023.0f);
				tmp = (unsigned int) x;
				tmp = (tmp * 0x00010001u) & 0xFF0000FFu;
				tmp = (tmp * 0x00000101u) & 0x0F00F00Fu;
				tmp = (tmp * 0x00000011u) & 0xC30C30C3u;
				tmp = (tmp * 0x00000005u) & 0x49249249u;
				unsigned int xx = tmp;

				tmp = (unsigned int) y;
				tmp = (tmp * 0x00010001u) & 0xFF0000FFu;
				tmp = (tmp * 0x00000101u) & 0x0F00F00Fu;
				tmp = (tmp * 0x00000011u) & 0xC30C30C3u;
				tmp = (tmp * 0x00000005u) & 0x49249249u;
				unsigned int yy = tmp;

				tmp = (unsigned int) z;
				tmp = (tmp * 0x00010001u) & 0xFF0000FFu;
				tmp = (tmp * 0x00000101u) & 0x0F00F00Fu;
				tmp = (tmp * 0x00000011u) & 0xC30C30C3u;
				tmp = (tmp * 0x00000005u) & 0x49249249u;
				unsigned int zz = tmp;
				uint up1 = xx * 4 + yy * 2 + zz;

				remapPos = p2.m_pos / m_dimension;
				tmp;
				x = std::min(std::max(remapPos.x * 1024.0f, 0.0f), 1023.0f);
				y = std::min(std::max(remapPos.y * 1024.0f, 0.0f), 1023.0f);
				z = std::min(std::max(remapPos.z * 1024.0f, 0.0f), 1023.0f);
				tmp = (unsigned int) x;
				tmp = (tmp * 0x00010001u) & 0xFF0000FFu;
				tmp = (tmp * 0x00000101u) & 0x0F00F00Fu;
				tmp = (tmp * 0x00000011u) & 0xC30C30C3u;
				tmp = (tmp * 0x00000005u) & 0x49249249u;
				xx = tmp;

				tmp = (unsigned int) y;
				tmp = (tmp * 0x00010001u) & 0xFF0000FFu;
				tmp = (tmp * 0x00000101u) & 0x0F00F00Fu;
				tmp = (tmp * 0x00000011u) & 0xC30C30C3u;
				tmp = (tmp * 0x00000005u) & 0x49249249u;
				yy = tmp;

				tmp = (unsigned int) z;
				tmp = (tmp * 0x00010001u) & 0xFF0000FFu;
				tmp = (tmp * 0x00000101u) & 0x0F00F00Fu;
				tmp = (tmp * 0x00000011u) & 0xC30C30C3u;
				tmp = (tmp * 0x00000005u) & 0x49249249u;
				zz = tmp;
				uint up2 = xx * 4 + yy * 2 + zz;

				return (up1 < up2);
			}

		private:
			// Default assignment operator to silence warning C4512.
			zOrderCmp& operator=(const zOrderCmp&);
			float3 m_dimension;
		};

	std::sort(sortedPoints.begin(), sortedPoints.end(), zOrderCmp(minValue->m_pos, maxValue->m_pos));
	}

}
