#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, random, heapq, ctypes
from collections import defaultdict
from pathlib import Path
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE, VIDEORESIZE, QUIT, KEYDOWN
from pyproj import Transformer
import xml.etree.ElementTree as ET
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram

# -------------------------------------------------------
# Paths y constantes
# -------------------------------------------------------
HERE = Path(__file__).resolve().parent
WINDOW_W, WINDOW_H = 1280, 800
OSM_FILE = HERE / "map-2.osm"
SHADERS_DIR = HERE / "shaders"
VERT_PATH = SHADERS_DIR / "lit.vert"
FRAG_PATH = SHADERS_DIR / "lit.frag"

ROAD_WIDTH_METERS = 10.0
FOV_DEG = 60.0
Z_NEAR, Z_FAR = 0.1, 5000.0

# Vehículos
MAX_VEHICLES = 400
VEHICLE_SPAWN_BURST = 20
CAR_SIZE = (4.2, 1.5, 2.0)   # largo, alto, ancho (m)
CAR_SPEED_MIN = 7.0          # m/s (~25 km/h)
CAR_SPEED_MAX = 16.0         # m/s (~58 km/h)

# Semáforos
TL_RED = 12.0
TL_YELLOW = 3.0
TL_GREEN = 12.0
TL_PERIOD = TL_RED + TL_YELLOW + TL_GREEN
TL_STOP_MARGIN = 1.8
TL_DETECTION_WIDTH = 2.5

# -------------------------------------------------------
# Math utils
# -------------------------------------------------------
def perspective(fov_deg, aspect, znear, zfar):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0], M[1, 1] = f / aspect, f
    M[2, 2] = (zfar + znear) / (znear - zfar)
    M[2, 3] = (2 * zfar * znear) / (znear - zfar)
    M[3, 2] = -1.0
    return M

def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def look_at(eye, center, up=(0, 1, 0)):
    eye, center, up = map(lambda a: np.array(a, dtype=np.float32), (eye, center, up))
    f = normalize(center - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0,:3], M[1,:3], M[2,:3] = s, u, -f
    T = np.eye(4, dtype=np.float32); T[:3,3] = -eye
    return M @ T

def translate(tx, ty, tz):
    M = np.eye(4, dtype=np.float32)
    M[0,3], M[1,3], M[2,3] = tx, ty, tz
    return M

def scale(sx, sy, sz):
    M = np.eye(4, dtype=np.float32)
    M[0,0], M[1,1], M[2,2] = sx, sy, sz
    return M

def rotate_y(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    M = np.eye(4, dtype=np.float32)
    M[0,0], M[0,2] = c,  s
    M[2,0], M[2,2] = -s, c
    return M

# -------------------------------------------------------
# OSM: carreteras
# -------------------------------------------------------
def load_osm_highways(osm_path: Path):
    root = ET.parse(str(osm_path)).getroot()
    nodes = {int(n.attrib["id"]): (float(n.attrib["lat"]), float(n.attrib["lon"])) for n in root.findall("node")}
    ways = []
    for way in root.findall("way"):
        if any(t.attrib["k"] == "highway" for t in way.findall("tag")):
            ways.append([int(nd.attrib["ref"]) for nd in way.findall("nd")])
    return nodes, ways

def local_transformer(nodes):
    lats, lons = zip(*nodes.values())
    lat0, lon0 = np.mean(lats), np.mean(lons)
    return Transformer.from_crs(
        "epsg:4326",
        f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
        always_xy=True,
    )

def project_nodes(nodes, transformer):
    return {nid: transformer.transform(lon, lat) for nid, (lat, lon) in nodes.items()}

def build_road_mesh(nodes_xy, ways, width_m=8.0):
    half_w = width_m / 2.0
    vertices, indices, base_index = [], [], 0
    for nds in ways:
        pts = [nodes_xy[n] for n in nds if n in nodes_xy]
        if len(pts) < 2: continue
        for i in range(len(pts) - 1):
            (x0, z0), (x1, z1) = pts[i], pts[i + 1]
            dx, dz = x1 - x0, z1 - z0
            seg_len = math.hypot(dx, dz)
            if seg_len < 1e-3: continue
            nx, nz = -dz / seg_len, dx / seg_len
            v0 = (x0 + nx * half_w, 0, z0 + nz * half_w)
            v1 = (x0 - nx * half_w, 0, z0 - nz * half_w)
            v2 = (x1 + nx * half_w, 0, z1 + nz * half_w)
            v3 = (x1 - nx * half_w, 0, z1 - nz * half_w)
            vertices += [*v0, *v1, *v2, *v3]
            indices  += [base_index, base_index+1, base_index+2,
                         base_index+2, base_index+1, base_index+3]
            base_index += 4
    return np.array(vertices, np.float32), np.array(indices, np.uint32)

# -------------------------------------------------------
# OSM: edificios
# -------------------------------------------------------
def parse_buildings_from_osm(osm_path: Path, transformer):
    root = ET.parse(str(osm_path)).getroot()
    latlon = {int(n.attrib["id"]): (float(n.attrib["lat"]), float(n.attrib["lon"])) for n in root.findall("node")}
    buildings = []
    for way in root.findall("way"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.findall("tag")}
        if "building" not in tags: continue
        nds = [int(nd.attrib["ref"]) for nd in way.findall("nd")]
        if len(nds) < 3: continue
        pts = []
        for nid in nds:
            if nid not in latlon: continue
            lat, lon = latlon[nid]
            x, z = transformer.transform(lon, lat)
            pts.append((x, z))
        if len(pts) < 3: continue
        if pts[0] != pts[-1]: pts.append(pts[0])
        h = 9.0
        if "height" in tags:
            s = tags["height"].lower().replace("m","").strip()
            try: h = float(s)
            except: pass
        elif "building:levels" in tags:
            try: h = float(tags["building:levels"]) * 3.0
            except: pass
        if abs(polygon_area(pts)) < 1.0: continue
        buildings.append((pts, h))
    return buildings

def polygon_area(poly_closed):
    a = 0.0
    for i in range(len(poly_closed) - 1):
        x1, z1 = poly_closed[i]; x2, z2 = poly_closed[i + 1]
        a += x1 * z2 - x2 * z1
    return 0.5 * a

def ear_clip_triangulate(poly_closed):
    P = poly_closed[:-1] if poly_closed[0] == poly_closed[-1] else poly_closed[:]
    n = len(P)
    if n < 3: return []
    ccw = polygon_area(poly_closed) > 0
    V = list(range(n)); tris = []
    while len(V) > 2:
        ear_found = False
        for k in range(len(V)):
            i0, i1, i2 = V[k - 1], V[k], V[(k + 1) % len(V)]
            ax, az = P[i0]; bx, bz = P[i1]; cx, cz = P[i2]
            cross = (bx - ax)*(cz - az) - (bz - az)*(cx - ax)
            if (cross > 0) != ccw: continue
            tris += [i0, i1, i2]; V.pop(k); ear_found = True; break
        if not ear_found: break
    return tris

def extrude_building_mesh(poly_closed, h):
    verts, inds, base = [], [], 0
    # paredes
    for (x0, z0), (x1, z1) in zip(poly_closed[:-1], poly_closed[1:]):
        v0, v1, v2, v3 = (x0, 0, z0), (x1, 0, z1), (x1, h, z1), (x0, h, z0)
        verts += [*v0, *v1, *v2, *v3]
        inds  += [base, base+1, base+2, base+2, base+3, base]
        base += 4
    # techo
    roof_off = base
    for x, z in poly_closed[:-1]:
        verts += [x, h, z]
    inds += [roof_off + i for i in ear_clip_triangulate(poly_closed)]
    return np.array(verts, np.float32), np.array(inds, np.uint32)

def build_all_buildings_mesh(buildings):
    V, I, base = [], [], 0
    for poly, h in buildings:
        v, ind = extrude_building_mesh(poly, h)
        if len(v) == 0 or len(ind) == 0: continue
        V.append(v); I.append(ind + base); base += len(v)//3
    if not V: return np.zeros((0,), np.float32), np.zeros((0,), np.uint32)
    return np.concatenate(V), np.concatenate(I)

# -------------------------------------------------------
# Normales e iluminación
# -------------------------------------------------------
def compute_flat_normals(verts, inds):
    if len(verts) == 0 or len(inds) == 0:
        return np.zeros((0,3), np.float32)
    V = verts.reshape(-1,3)
    norms = np.zeros_like(V)
    for i0, i1, i2 in inds.reshape(-1,3):
        v0, v1, v2 = V[i0], V[i1], V[i2]
        n = np.cross(v1 - v0, v2 - v0)
        ln = np.linalg.norm(n)
        if ln > 1e-6: n /= ln
        norms[i0] += n; norms[i1] += n; norms[i2] += n
    lens = np.linalg.norm(norms, axis=1)
    lens[lens == 0] = 1.0
    norms /= lens[:,None]
    return norms.astype(np.float32)

# -------------------------------------------------------
# OpenGL helpers
# -------------------------------------------------------
def create_window_and_context():
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, 1)
    pygame.display.set_mode((WINDOW_W, WINDOW_H), DOUBLEBUF | OPENGL | RESIZABLE)
    pygame.display.set_caption("OSM 3D Viewer")

    print("GL_VENDOR:", glGetString(GL_VENDOR).decode())
    print("GL_RENDERER:", glGetString(GL_RENDERER).decode())
    print("GL_VERSION:", glGetString(GL_VERSION).decode())

    # VAO dummy necesario en macOS/Metal core profile
    dummy_vao = glGenVertexArrays(1)
    glBindVertexArray(dummy_vao)
    return dummy_vao

def load_shader_program(vp: Path, fp: Path):
    with open(vp) as f: vert = f.read()
    with open(fp) as f: frag = f.read()
    return compileProgram(compileShader(vert, GL_VERTEX_SHADER), compileShader(frag, GL_FRAGMENT_SHADER))

def build_vao_pos_norm(verts, norms, inds):
    inter = np.hstack([verts.reshape(-1,3), norms.reshape(-1,3)]).astype(np.float32).ravel()
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, inter.nbytes, inter, GL_STATIC_DRAW)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, inds.nbytes, inds, GL_STATIC_DRAW)

    stride = 24
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

    glBindVertexArray(0)
    return vao, vbo, ebo

# -------------------------------------------------------
# Grafo viario / rutas
# -------------------------------------------------------
def build_graph(nodes_xy, ways):
    adj = defaultdict(list)
    for nds in ways:
        for a, b in zip(nds[:-1], nds[1:]):
            if a not in nodes_xy or b not in nodes_xy: continue
            (xa, za), (xb, zb) = nodes_xy[a], nodes_xy[b]
            d = math.hypot(xb - xa, zb - za)
            if d < 0.5: continue
            adj[a].append((b, d))
            adj[b].append((a, d))
    return adj

def dijkstra_path(adj, start, goal):
    if start not in adj or goal not in adj: return []
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    seen = set()
    while pq:
        d,u = heapq.heappop(pq)
        if u in seen: continue
        seen.add(u)
        if u == goal: break
        for v,w in adj[u]:
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd; prev[v] = u; heapq.heappush(pq,(nd,v))
    if goal != start and goal not in prev: return []
    path = [goal]
    while path[-1] != start:
        path.append(prev.get(path[-1], start))
    path.reverse()
    return path

def path_points_from_nodes(nodes_xy, node_path):
    return [nodes_xy[n] for n in node_path if n in nodes_xy]

# -------------------------------------------------------
# Semáforos OSM
# -------------------------------------------------------
class TrafficLight:
    __slots__ = ("x","z","offset")
    def __init__(self, x, z, offset):
        self.x, self.z, self.offset = x, z, offset
    def phase(self, t):
        tau = (t + self.offset) % TL_PERIOD
        if tau < TL_GREEN: return "green"
        if tau < TL_GREEN + TL_YELLOW: return "yellow"
        return "red"
    def is_red(self, t): return self.phase(t) == "red"
    def is_green(self, t): return self.phase(t) == "green"

def parse_traffic_lights(osm_path: Path, transformer):
    root = ET.parse(str(osm_path)).getroot()
    tls = []
    for node in root.findall("node"):
        has_tl = any(tag.attrib.get("k") == "highway" and tag.attrib.get("v") == "traffic_signals"
                     for tag in node.findall("tag"))
        if not has_tl: continue
        lat = float(node.attrib["lat"]); lon = float(node.attrib["lon"])
        x, z = transformer.transform(lon, lat)
        off = (hash((round(x,1), round(z,1))) % 1000) * 0.01
        tls.append(TrafficLight(x, z, off))
    return tls

# -------------------------------------------------------
# Malla de cubo (coches + cabezas de semáforo)
# -------------------------------------------------------
def build_unit_cube_mesh():
    p = [
        # +X
        ([0.5,-0.5,-0.5],[1,0,0]), ([0.5,-0.5,0.5],[1,0,0]), ([0.5,0.5,0.5],[1,0,0]), ([0.5,0.5,-0.5],[1,0,0]),
        # -X
        ([-0.5,-0.5,0.5],[-1,0,0]), ([-0.5,-0.5,-0.5],[-1,0,0]), ([-0.5,0.5,-0.5],[-1,0,0]), ([-0.5,0.5,0.5],[-1,0,0]),
        # +Y
        ([-0.5,0.5,-0.5],[0,1,0]), ([0.5,0.5,-0.5],[0,1,0]), ([0.5,0.5,0.5],[0,1,0]), ([-0.5,0.5,0.5],[0,1,0]),
        # -Y
        ([-0.5,-0.5,0.5],[0,-1,0]), ([0.5,-0.5,0.5],[0,-1,0]), ([0.5,-0.5,-0.5],[0,-1,0]), ([-0.5,-0.5,-0.5],[0,-1,0]),
        # +Z
        ([-0.5,-0.5,0.5],[0,0,1]), ([0.5,-0.5,0.5],[0,0,1]), ([0.5,0.5,0.5],[0,0,1]), ([-0.5,0.5,0.5],[0,0,1]),
        # -Z
        ([0.5,-0.5,-0.5],[0,0,-1]), ([-0.5,-0.5,-0.5],[0,0,-1]), ([-0.5,0.5,-0.5],[0,0,-1]), ([0.5,0.5,-0.5],[0,0,-1]),
    ]
    verts = np.array([c[0] for c in p], np.float32).ravel()
    norms = np.array([c[1] for c in p], np.float32).ravel()
    idx = []
    for f in range(6):
        o = f*4
        idx += [o, o+1, o+2,  o, o+2, o+3]
    return verts, norms, np.array(idx, np.uint32)

# -------------------------------------------------------
# Vehículos
# -------------------------------------------------------
class Vehicle:
    __slots__ = ("pts","seg_index","t","speed","color","stopped")
    def __init__(self, pts, speed, color):
        self.pts = pts; self.seg_index = 0; self.t = 0.0
        self.speed = speed; self.color = color; self.stopped = False
    def _closest_light_on_segment(self, tlights, tol_width=TL_DETECTION_WIDTH):
        if self.seg_index >= len(self.pts) - 1: return None
        (x0,z0) = self.pts[self.seg_index]; (x1,z1) = self.pts[self.seg_index+1]
        vx, vz = (x1-x0), (z1-z0); seg_len2 = vx*vx + vz*vz
        if seg_len2 < 1e-6: return None
        best = None
        for L in tlights:
            wx, wz = L.x - x0, L.z - z0
            u = (wx*vx + wz*vz) / seg_len2
            if u < 0.0 or u > 1.0: continue
            px, pz = x0 + u*vx, z0 + u*vz
            dlat = math.hypot(L.x - px, L.z - pz)
            if dlat <= tol_width:
                if best is None or u < best[0]: best = (u, L)
        return best
    def update(self, dt, tlights, sim_time):
        if self.seg_index >= len(self.pts) - 1: return False
        (x0,z0) = self.pts[self.seg_index]; (x1,z1) = self.pts[self.seg_index+1]
        seg_len = math.hypot(x1-x0, z1-z0)
        if seg_len < 1e-3:
            self.seg_index += 1; self.t = 0.0; return True
        hit = self._closest_light_on_segment(tlights)
        if hit is not None:
            u_light, light = hit
            dist_to_light = max(0.0, (u_light - max(self.t,0.0)) * seg_len)
            if light.is_red(sim_time) and dist_to_light <= TL_STOP_MARGIN:
                self.stopped = True; return True
            if self.stopped and light.is_green(sim_time):
                self.stopped = False
        if self.stopped: return True
        self.t += (self.speed * dt) / seg_len
        while self.t >= 1.0 and self.seg_index < len(self.pts)-1:
            self.t -= 1.0; self.seg_index += 1
            if self.seg_index >= len(self.pts)-1:
                self.t = 0.0; break
        return True
    def pose(self):
        if self.seg_index >= len(self.pts) - 1:
            i = max(0, len(self.pts)-2)
            (x0,z0),(x1,z1) = self.pts[i], self.pts[i+1]
            return x1, z1, math.atan2(z1-z0, x1-x0)
        (x0,z0) = self.pts[self.seg_index]; (x1,z1) = self.pts[self.seg_index+1]
        x = x0 + (x1-x0)*self.t; z = z0 + (z1-z0)*self.t
        yaw = math.atan2(z1-z0, x1-x0)
        return x, z, yaw

# -------------------------------------------------------
# Escena principal
# -------------------------------------------------------
def main():
    # ventana/GL
    dummy_vao = create_window_and_context()
    glEnable(GL_DEPTH_TEST)
    shader = load_shader_program(VERT_PATH, FRAG_PATH)
    glUseProgram(shader)

    # OSM → mundo
    nodes, ways = load_osm_highways(OSM_FILE)
    transformer = local_transformer(nodes)
    nodes_xy = project_nodes(nodes, transformer)
    v_road, i_road = build_road_mesh(nodes_xy, ways, ROAD_WIDTH_METERS)
    buildings = parse_buildings_from_osm(OSM_FILE, transformer)
    v_build, i_build = build_all_buildings_mesh(buildings)
    traffic_lights = parse_traffic_lights(OSM_FILE, transformer)
    print(f"Semáforos detectados: {len(traffic_lights)}")

    # Normales y VAOs
    n_road = compute_flat_normals(v_road, i_road)
    n_build = compute_flat_normals(v_build, i_build)
    vao_r, vbo_r, ebo_r = build_vao_pos_norm(v_road, n_road, i_road)
    vao_b = vbo_b = ebo_b = None
    if len(v_build) > 0:
        vao_b, vbo_b, ebo_b = build_vao_pos_norm(v_build, n_build, i_build)
    cube_v, cube_n, cube_i = build_unit_cube_mesh()
    vao_cube, vbo_cube, ebo_cube = build_vao_pos_norm(cube_v, cube_n, cube_i)

    # Uniforms comunes
    xs, zs = zip(*nodes_xy.values())
    world_center = (-float(np.mean(xs)), 0.0, -float(np.mean(zs)))
    world_model = translate(*world_center)

    loc_mvp = glGetUniformLocation(shader, "uMVP")
    loc_color = glGetUniformLocation(shader, "uColor")
    loc_model = glGetUniformLocation(shader, "uModel")
    loc_normalmat = glGetUniformLocation(shader, "uNormalMat")
    loc_lightdir = glGetUniformLocation(shader, "uLightDir")

    # Cámara FPS (ratón absoluto + teclado)
    eye = np.array([0.0, 60.0, 150.0], np.float32)
    cam_front = np.array([0.0, -0.3, -1.0], np.float32)
    cam_up = np.array([0.0, 1.0, 0.0], np.float32)
    yaw, pitch = -90.0, -12.0
    mouse_sens, move_speed = 0.12, 40.0
    proj = perspective(FOV_DEG, WINDOW_W / WINDOW_H, Z_NEAR, Z_FAR)

    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    pygame.mouse.set_pos((WINDOW_W // 2, WINDOW_H // 2))

    # Grafo/rutas
    graph = build_graph(nodes_xy, ways)
    node_ids = [n for n in graph.keys()]
    rng = random.Random(42)

    def random_route_points():
        if not node_ids: return []
        s = rng.choice(node_ids); t = rng.choice(node_ids)
        path_nodes = dijkstra_path(graph, s, t)
        if len(path_nodes) < 2: return []
        return path_points_from_nodes(nodes_xy, path_nodes)

    # Vehículos
    vehicles = []
    def spawn_burst(k=VEHICLE_SPAWN_BURST):
        nonlocal vehicles
        for _ in range(k):
            pts = random_route_points()
            if len(pts) < 2: continue
            speed = rng.uniform(CAR_SPEED_MIN, CAR_SPEED_MAX)
            color = (rng.uniform(0.6,1.0), rng.uniform(0.2,0.9), rng.uniform(0.2,0.9))
            vehicles.append(Vehicle(pts, speed, color))
            if len(vehicles) >= MAX_VEHICLES: break
    spawn_burst(VEHICLE_SPAWN_BURST)

    clock = pygame.time.Clock()
    running, wire = True, False
    sim_time = 0.0

    while running:
        dt = clock.tick(60) / 1000.0
        sim_time += dt

        for e in pygame.event.get():
            if e.type == QUIT: running = False
            elif e.type == VIDEORESIZE:
                w, h = max(1, e.w), max(1, e.h)
                glViewport(0, 0, w, h)
                proj = perspective(FOV_DEG, w / h, Z_NEAR, Z_FAR)
                pygame.mouse.set_pos((w // 2, h // 2))
            elif e.type == KEYDOWN:
                if e.key == pygame.K_ESCAPE: running = False
                elif e.key == pygame.K_l:
                    wire = not wire
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if wire else GL_FILL)
                elif e.key == pygame.K_v:
                    spawn_burst(VEHICLE_SPAWN_BURST)
                elif e.key == pygame.K_c:
                    vehicles.clear()

        # --------- RATÓN absoluto respecto al centro (macOS-safe) ---------
        if pygame.mouse.get_focused():
            w, h = pygame.display.get_window_size()
            cx, cy = w // 2, h // 2
            x, y = pygame.mouse.get_pos()
            mx, my = (x - cx), (y - cy)
        else:
            mx, my = 0, 0

        yaw   += mx * mouse_sens
        pitch -= my * mouse_sens
        pitch = max(-89.0, min(89.0, pitch))
        yaw_r, pitch_r = math.radians(yaw), math.radians(pitch)
        cam_front = normalize(np.array([
            math.cos(yaw_r) * math.cos(pitch_r),
            math.sin(pitch_r),
            math.sin(yaw_r) * math.cos(pitch_r)
        ], np.float32))
        # ------------------------------------------------------------------

        # Teclado: rotación adicional (garantiza yaw/pitch)
        keys = pygame.key.get_pressed()
        key_yaw_speed = 90.0     # grados/seg
        key_pitch_speed = 70.0   # grados/seg
        if keys[pygame.K_LEFT]:  yaw -= key_yaw_speed * dt
        if keys[pygame.K_RIGHT]: yaw += key_yaw_speed * dt
        if keys[pygame.K_UP]:    pitch += key_pitch_speed * dt
        if keys[pygame.K_DOWN]:  pitch -= key_pitch_speed * dt
        pitch = max(-89.0, min(89.0, pitch))

        # Movimiento cámara
        speed = move_speed * dt
        right = normalize(np.cross(cam_front, cam_up))
        if keys[pygame.K_w]: eye += cam_front * speed
        if keys[pygame.K_s]: eye -= cam_front * speed
        if keys[pygame.K_a]: eye -= right * speed
        if keys[pygame.K_d]: eye += right * speed
        if keys[pygame.K_SPACE]: eye[1] += speed
        if keys[pygame.K_LSHIFT]: eye[1] -= speed

        # Update vehículos (con semáforos)
        alive = []
        for v in vehicles:
            v.update(dt, traffic_lights, sim_time)
            if v.seg_index >= len(v.pts) - 1:
                pts = random_route_points()
                if len(pts) >= 2:
                    v.pts = pts; v.seg_index = 0; v.t = 0.0; v.stopped = False
                    alive.append(v)
            else:
                alive.append(v)
        vehicles = alive[:MAX_VEHICLES]

        # Render
        glClearColor(0.05, 0.07, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = look_at(eye, eye + cam_front, cam_up)
        light_dir = np.array([0.3, 1.0, 0.5], np.float32)

        glUseProgram(shader)
        glUniform3f(loc_lightdir, *light_dir)

        # Carreteras
        m = world_model
        mvp = (proj @ view @ m).astype(np.float32)
        normalmat = np.linalg.inv(m[:3,:3]).T.astype(np.float32)
        glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp.T)
        glUniformMatrix4fv(loc_model, 1, GL_FALSE, m.T)
        glUniformMatrix3fv(loc_normalmat, 1, GL_FALSE, normalmat.T)
        glUniform3f(loc_color, 0.1, 0.9, 0.9)
        glBindVertexArray(vao_r)
        glDrawElements(GL_TRIANGLES, len(i_road), GL_UNSIGNED_INT, None)

        # Edificios
        if vao_b is not None:
            glUniform3f(loc_color, 0.8, 0.8, 0.82)
            glBindVertexArray(vao_b)
            glDrawElements(GL_TRIANGLES, len(i_build), GL_UNSIGNED_INT, None)

        # Semáforos (poste + cabeza)
        glBindVertexArray(vao_cube)
        for L in traffic_lights:
            pole = translate(L.x, 1.5, L.z) @ scale(0.15, 3.0, 0.15)
            wm = world_model @ pole
            mvp = (proj @ view @ wm).astype(np.float32)
            normalmat = np.linalg.inv(wm[:3,:3]).T.astype(np.float32)
            glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp.T)
            glUniformMatrix4fv(loc_model, 1, GL_FALSE, wm.T)
            glUniformMatrix3fv(loc_normalmat, 1, GL_FALSE, normalmat.T)
            glUniform3f(loc_color, 0.2, 0.2, 0.2)
            glDrawElements(GL_TRIANGLES, len(cube_i), GL_UNSIGNED_INT, None)

            is_green = True if not L.is_red(sim_time) else False
            head_col = (0.15, 0.85, 0.25) if is_green else (0.9, 0.2, 0.2)
            head = translate(L.x, 3.2, L.z) @ scale(0.35, 0.35, 0.35)
            wm = world_model @ head
            mvp = (proj @ view @ wm).astype(np.float32)
            normalmat = np.linalg.inv(wm[:3,:3]).T.astype(np.float32)
            glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp.T)
            glUniformMatrix4fv(loc_model, 1, GL_FALSE, wm.T)
            glUniformMatrix3fv(loc_normalmat, 1, GL_FALSE, normalmat.T)
            glUniform3f(loc_color, *head_col)
            glDrawElements(GL_TRIANGLES, len(cube_i), GL_UNSIGNED_INT, None)

        # Vehículos
        Lg, Hg, Wg = CAR_SIZE
        sm = scale(Lg, Hg, Wg)
        for v in vehicles:
            x, z, yaw_obj = v.pose()
            obj = translate(x, Hg*0.5, z) @ rotate_y(yaw_obj) @ sm
            wm = world_model @ obj
            mvp = (proj @ view @ wm).astype(np.float32)
            normalmat = np.linalg.inv(wm[:3,:3]).T.astype(np.float32)
            glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp.T)
            glUniformMatrix4fv(loc_model, 1, GL_FALSE, wm.T)
            glUniformMatrix3fv(loc_normalmat, 1, GL_FALSE, normalmat.T)
            glUniform3f(loc_color, *v.color)
            glBindVertexArray(vao_cube)
            glDrawElements(GL_TRIANGLES, len(cube_i), GL_UNSIGNED_INT, None)

        pygame.display.flip()

        # Recentramos al final del frame (clave en macOS)
        if pygame.mouse.get_focused():
            w, h = pygame.display.get_window_size()
            pygame.mouse.set_pos((w // 2, h // 2))

    # Limpieza
    glDeleteVertexArrays(1, [vao_r]); glDeleteBuffers(1, [vbo_r]); glDeleteBuffers(1, [ebo_r])
    if vao_b:
        glDeleteVertexArrays(1, [vao_b]); glDeleteBuffers(1, [vbo_b]); glDeleteBuffers(1, [ebo_b])
    glDeleteVertexArrays(1, [vao_cube]); glDeleteBuffers(1, [vbo_cube]); glDeleteBuffers(1, [ebo_cube])
    glDeleteVertexArrays(1, [dummy_vao])
    glDeleteProgram(shader)
    pygame.quit()

if __name__ == "__main__":
    main()
