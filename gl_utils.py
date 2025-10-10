import numpy as np
from OpenGL.GL import *
import ctypes

def create_vao(vertices: np.ndarray, indices: np.ndarray):
    """
    vertices: flat float32 array len = N*3
    indices : uint32 array
    returns : vao, vbo, ebo, count (int)
    """
    # 1) Enforce tipos/contigüidad
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1)
    indices  = np.asarray(indices,  dtype=np.uint32).reshape(-1)

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)

    glBindVertexArray(vao)

    # VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # EBO (debe hacerse con el VAO ligado para que quede asociado)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Atributo posición en location=0 (3 floats, stride 12 bytes)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))

    glBindVertexArray(0)

    return vao, vbo, ebo, int(indices.size)


def translate(tx, ty, tz):
    m = np.eye(4, dtype=np.float32)
    m[3, 0:3] = [tx, ty, tz]
    return m

def scale(sx, sy, sz):
    m = np.eye(4, dtype=np.float32)
    m[0,0] = sx
    m[1,1] = sy
    m[2,2] = sz
    return m

def rotate_y(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    m = np.eye(4, dtype=np.float32)
    m[0,0] = c; m[0,2] = s
    m[2,0] = -s; m[2,2] = c
    return m

def perspective(fovy, aspect, near, far):
    f = 1.0 / np.tan(fovy / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)  # término de traslación en Z
    m[3, 2] = -1.0                               # -1 en la fila inferior, col Z
    # m[3,3] = 0 por defecto
    return m


def look_at(eye, center, up):
    f = np.array(center, dtype=np.float32) - np.array(eye, dtype=np.float32)
    f = f / np.linalg.norm(f)
    u = np.array(up, dtype=np.float32)
    u = u / np.linalg.norm(u)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0,0:3] = s
    m[1,0:3] = u
    m[2,0:3] = -f
    translate_mat = np.eye(4, dtype=np.float32)
    translate_mat[3,0:3] = -np.array(eye, dtype=np.float32)
    return m @ translate_mat