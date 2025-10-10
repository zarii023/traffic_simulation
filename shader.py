from OpenGL.GL import *

def load_shader_source(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def _compile_single_shader(src: str, shader_type) -> int:
    sh = glCreateShader(shader_type)
    glShaderSource(sh, src)
    glCompileShader(sh)
    if glGetShaderiv(sh, GL_COMPILE_STATUS) != GL_TRUE:
        log = glGetShaderInfoLog(sh).decode('utf-8', errors='ignore')
        kind = 'VERTEX' if shader_type == GL_VERTEX_SHADER else 'FRAGMENT'
        glDeleteShader(sh)
        raise RuntimeError(f"{kind} SHADER COMPILE ERROR:\n{log}")
    return sh

def compile_shader(vertex_path, fragment_path) -> int:
    vert = _compile_single_shader(load_shader_source(vertex_path), GL_VERTEX_SHADER)
    frag = _compile_single_shader(load_shader_source(fragment_path), GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vert); glAttachShader(prog, frag)
    glLinkProgram(prog)
    glDeleteShader(vert); glDeleteShader(frag)
    if glGetProgramiv(prog, GL_LINK_STATUS) != GL_TRUE:
        log = glGetProgramInfoLog(prog).decode('utf-8', errors='ignore')
        glDeleteProgram(prog)
        raise RuntimeError(f"PROGRAM LINK ERROR:\n{log}")
    return prog
