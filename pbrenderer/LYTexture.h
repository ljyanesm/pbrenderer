#ifndef TEXTURE_H
#define	TEXTURE_H

#include <string>

#include <GL/glew.h>
#include <SDL.h>
#undef main
#include <SDL_image.h>

class LYTexture
{
public:
    LYTexture(GLenum TextureTarget, const char *FileName);

    bool Load();

    void Bind(GLenum TextureUnit);

private:
    const char *m_fileName;
    GLenum m_textureTarget;
    GLuint m_textureObj;
};


#endif	/* TEXTURE_H */

