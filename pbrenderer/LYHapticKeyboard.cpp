#include "LYHapticKeyboard.h"


LYHapticKeyboard::LYHapticKeyboard(void)
{
}


LYHapticKeyboard::~LYHapticKeyboard(void)
{
}

glm::vec3 LYHapticKeyboard::getPosition(){
	return m_position;
}
void LYHapticKeyboard::setPosition(glm::vec3 pos){

}
glm::vec3 LYHapticKeyboard::getForceFeedback(){
	return glm::vec3();
}

float LYHapticKeyboard::getSpeed()
{
	return m_speed;
}