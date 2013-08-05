#include <iostream>
#include "texture.h"

Texture::Texture(GLenum TextureTarget, const char* FileName)
{
    m_textureTarget = TextureTarget;
    m_fileName      = FileName;
}

bool Texture::Load()
{
	SDL_Surface *img = IMG_Load(m_fileName);
	SDL_PixelFormat form = {NULL, 32, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0xff000000, 0x00ff0000, 0x0000ff00, 0x000000ff, 0, 255};
	SDL_Surface *img2 = SDL_ConvertSurface(img, &form, SDL_SWSURFACE);

    glGenTextures(1, &m_textureObj);
    glBindTexture(m_textureTarget, m_textureObj);
	glTexParameteri(m_textureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(m_textureTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img->w, img->h, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, img2->pixels);

    return true;
}

void Texture::Bind(GLenum TextureUnit)
{
    glActiveTexture(TextureUnit);
    glBindTexture(m_textureTarget, m_textureObj);
}
