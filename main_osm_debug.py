#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
from pathlib import Path
import ctypes

import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE, VIDEORESIZE, QUIT, KEYDOWN
from pyproj import Transformer
import xml.etree.ElementTree as ET

from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram


# ---------------------------------------------------------------------------
# Paths y constantes
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
WINDOW_W, WINDOW_H = 1280, 800

OSM_FILE = HERE / "map-2.osm"               # cambia si quieres otro .osm
SHADERS_DIR = HERE / "shaders"
VERT_PATH = SHADERS_DIR / "basic.vert"
FRAG_PATH = SHADERS_DIR / "basic.frag"

ROAD_WIDTH_METERS = 10.0                     # ancho de calzada (prueba 8–12 m)
FOV_DEG = 60.0
Z_NEAR = 0.1
Z_FAR  = 5000.0

# ---------------------------------------------------------------------------
# Utilidades matemáticas
# ---------------------------------------------------------------------------
def perspective(fov_deg, aspect, znear, zfar):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (zfar + znear) / (znear - zfar)
    M[2, 3] = (2 * zfar * znear) / (znear - zfar)
    M[3, 2] = -1.0
    return M

def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def look_at(eye, center, up=(0, 1, 0)):
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    f = normalize(center - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f

    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    return M @ T

def translate(tx, ty, tz):
    M = np.eye(4, dtype=np.float32)
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M

# ---------------------------------------------------------------------------
# Carga y parseo OSM
# ---------------------------------------------------------------------------
def load_osm_highways(osm_path: Path):
    if not osm_path.exists():
        raise FileNotFoundError(f"No se encontró el OSM: {osm_path}")

    tree = ET.parse(str(osm_path))
    root = tree.getroot()

    nodes = {}
    for node in root.findall("node"):
        nid = int(node.attrib["id"])
        lat = float(node.attrib["lat"])
        lon = float(node.attrib["lon"])
        nodes[nid] = (lat, lon)

    ways = []
    for way in root.findall("way"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.findall("tag")}
        if "highway" in tags:
            nds = [int(nd.attrib["ref"]) for nd in way.findall("nd")]
            ways.append(nds)

    return nodes, ways

def local_transformer(nodes):
    lats = [lat for (lat, lon) in nodes.values()]
    lons = [lon for (lat, lon) in nodes.values()]
    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    # Proyección tmerc local centrada en el área
    transformer = Transformer.from_crs(
        "epsg:4326",
        f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
        always_xy=True,
    )
    return transformer

def project_nodes(nodes, transformer):
    xy = {}
    for nid, (lat, lon) in nodes.items():
        x, z = transformer.transform(lon, lat)
        xy[nid] = (x, z)
    return xy

# ---------------------------------------------------------------------------
# Generación de malla: tira de triángulos a lo largo de cada polilínea
# ---------------------------------------------------------------------------
def build_road_mesh(nodes_xy, ways, width_m=8.0):
    half_w = width_m / 2.0
    vertices = []
    indices = []
    base_index = 0
    total_segments = 0

    for nds in ways:
        pts = [nodes_xy[n] for n in nds if n in nodes_xy]
        if len(pts) < 2:
            continue
        # Para cada segmento generamos 2 triángulos (4 vértices compartidos por segmento)
        for i in range(len(pts) - 1):
            (x0, z0) = pts[i]
            (x1, z1) = pts[i + 1]
            dx, dz = (x1 - x0, z1 - z0)
            seg_len = math.hypot(dx, dz)
            if seg_len < 1e-3:
                continue
            nx, nz = (-dz / seg_len, dx / seg_len)  # normal lateral en el plano XZ

            # 4 vértices del quad (Y=0 por ahora)
            v0 = (x0 + nx * half_w, 0.0, z0 + nz * half_w)
            v1 = (x0 - nx * half_w, 0.0, z0 - nz * half_w)
            v2 = (x1 + nx * half_w, 0.0, z1 + nz * half_w)
            v3 = (x1 - nx * half_w, 0.0, z1 - nz * half_w)

            vertices.extend(v0); vertices.extend(v1); vertices.extend(v2); vertices.extend(v3)

            # Dos triángulos: (0,1,2) y (2,1,3) relativos al segmento
            indices.extend([base_index + 0, base_index + 1, base_index + 2,
                            base_index + 2, base_index + 1, base_index + 3])
            base_index += 4
            total_segments += 1

    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    return vertices, indices, total_segments

# ---------------------------------------------------------------------------
# OpenGL helpers
# ---------------------------------------------------------------------------
def create_window_and_context():
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, 1)
    pygame.display.set_mode((WINDOW_W, WINDOW_H), DOUBLEBUF | OPENGL | RESIZABLE)
    pygame.display.set_caption("OSM 3D Debug Viewer")

    # Diagnóstico del contexto
    print("GL_VENDOR   :", glGetString(GL_VENDOR).decode())
    print("GL_RENDERER :", glGetString(GL_RENDERER).decode())
    print("GL_VERSION  :", glGetString(GL_VERSION).decode())
    print("GLSL        :", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())

def load_shader_program(vert_path: Path, frag_path: Path):
    with open(vert_path, "r", encoding="utf-8") as f:
        vert_src = f.read()
    with open(frag_path, "r", encoding="utf-8") as f:
        frag_src = f.read()

    try:
        program = compileProgram(
            compileShader(vert_src, GL_VERTEX_SHADER),
            compileShader(frag_src, GL_FRAGMENT_SHADER)
        )
    except Exception as e:
        print("Error compilando/enlazando shaders:", e)
        raise
    return program

def build_vao(vertices: np.ndarray, indices: np.ndarray):
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)  # importante: ligado con VAO
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # layout(location = 0) in vec3 aPos;
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

    glBindVertexArray(0)
    return vao, vbo, ebo

# ---------------------------------------------------------------------------
# Escena
# ---------------------------------------------------------------------------
def main():
    # 1) Ventana + contexto
    create_window_and_context()
    glViewport(0, 0, WINDOW_W, WINDOW_H)
    glEnable(GL_DEPTH_TEST)
    # --- después de create_window_and_context()
    glViewport(0, 0, WINDOW_W, WINDOW_H)
    glEnable(GL_DEPTH_TEST)

    # 👇 VAO mínimo para contentar al core profile de macOS
    dummy_vao = glGenVertexArrays(1)
    glBindVertexArray(dummy_vao)

    # 2) Shaders
    if not VERT_PATH.exists() or not FRAG_PATH.exists():
        raise RuntimeError(f"No se encuentran shaders en {SHADERS_DIR}")
    shader = load_shader_program(VERT_PATH, FRAG_PATH)
    glUseProgram(shader)

    # 3) Carga OSM -> proyección -> malla
    nodes, ways = load_osm_highways(OSM_FILE)
    transformer = local_transformer(nodes)
    nodes_xy = project_nodes(nodes, transformer)
    vertices, indices, segs = build_road_mesh(nodes_xy, ways, width_m=ROAD_WIDTH_METERS)
    print(f"ways={len(ways)}, segments={segs}, vertices={len(vertices)//3}, indices={len(indices)}")

    # Centro geométrico para colocar la cámara mirando al origen
    xs = [xy[0] for xy in nodes_xy.values()]
    zs = [xy[1] for xy in nodes_xy.values()]
    cx = float(np.mean(xs))
    cz = float(np.mean(zs))
    model = translate(-cx, 0.0, -cz)

    vao, vbo, ebo = build_vao(vertices, indices)
    index_count = indices.size

    # 4) Triángulo fusible (para depurar pipeline rápidamente)
    #tri_v = np.array([[-50, 0, -50], [50, 0, -50], [0, 0, 50]], dtype=np.float32).ravel()
    #tri_i = np.array([0, 1, 2], dtype=np.uint32)
    #tri_vao, tri_vbo, tri_ebo = build_vao(tri_v, tri_i)
    #tri_count = tri_i.size

    # 5) Uniforms
    loc_mvp = glGetUniformLocation(shader, "uMVP")
    loc_color = glGetUniformLocation(shader, "uColor")
    if loc_mvp == -1 or loc_color == -1:
        raise RuntimeError(f"Uniform no encontrado (uMVP={loc_mvp}, uColor={loc_color}). ¿GLSL correcto?")

    # 6) Cámara
    eye = np.array([0.0, 180.0, 180.0], dtype=np.float32)
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    aspect = WINDOW_W / WINDOW_H
    proj = perspective(FOV_DEG, aspect, Z_NEAR, Z_FAR)

    running = True
    wire = False
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                w, h = event.w, event.h
                if h == 0: h = 1
                pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL | RESIZABLE)
                glViewport(0, 0, w, h)
                aspect = w / h
                proj = perspective(FOV_DEG, aspect, Z_NEAR, Z_FAR)
            elif event.type == KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_w:
                    wire = not wire
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if wire else GL_FILL)
                elif event.key == pygame.K_q:   # acercar
                    eye *= 0.9
                elif event.key == pygame.K_e:   # alejar
                    eye *= 1.1

        glClearColor(0.05, 0.07, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = look_at(eye, center, up)
        mvp = (proj @ view @ model).astype(np.float32)

        glUseProgram(shader)
        glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp.T)   # NumPy filas → .T con transpose=False

        # Triángulo fusible (amarillo)
        #glUniform3f(loc_color, 1.0, 1.0, 0.2)
        #glBindVertexArray(tri_vao)
        #glDrawElements(GL_TRIANGLES, tri_count, GL_UNSIGNED_INT, None)

        # Carreteras (cian)
        glUniform3f(loc_color, 0.1, 0.9, 0.9)
        glBindVertexArray(vao)
        glDeleteVertexArrays(1, [dummy_vao])

        glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, None)

        glBindVertexArray(0)
        pygame.display.flip()
        clock.tick(60)

    # Limpieza
    glDeleteVertexArrays(1, [vao]); glDeleteBuffers(1, [vbo]); glDeleteBuffers(1, [ebo])
    #glDeleteVertexArrays(1, [tri_vao]); glDeleteBuffers(1, [tri_vbo]); glDeleteBuffers(1, [tri_ebo])
    glDeleteProgram(shader)
    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        pygame.quit()
        sys.exit(1)
