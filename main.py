import numpy as np
from math import cos, sin, radians
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from PIL import Image, ImageTk


def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def T(dx, dy, dz):
    M = np.eye(4); M[:3, 3] = [dx, dy, dz]; return M

def S(sx, sy, sz):
    M = np.eye(4); M[0,0],M[1,1],M[2,2]=sx,sy,sz; return M

def Rx(a):
    a=radians(a); ca,sa=cos(a),sin(a)
    M=np.eye(4); M[1,1],M[1,2],M[2,1],M[2,2]=ca,-sa,sa,ca; return M

def Ry(a):
    a=radians(a); ca,sa=cos(a),sin(a)
    M=np.eye(4); M[0,0],M[0,2],M[2,0],M[2,2]=ca,sa,-sa,ca; return M

def Rz(a):
    a=radians(a); ca,sa=cos(a),sin(a)
    M=np.eye(4); M[0,0],M[0,1],M[1,0],M[1,1]=ca,-sa,sa,ca; return M


@dataclass
class Face:
    idx: tuple    # индексы вершин (треугольник)
    uv:  tuple    # соответствующие UV (по 3 пары)
    normal: np.ndarray = None  # нормаль грани (3,)

class Mesh:
    def __init__(self, V, F, UV=None, VN=None):
        self.V = np.asarray(V, float)
        self.F = []
        if UV is None:
            UV = np.zeros((len(V), 2), float)
        self.UV = np.asarray(UV, float)
        self.VN = VN.copy() if VN is not None else None

        for tri in F:
            # если нам уже дали Face с uv — просто принять
            if isinstance(tri, Face):
                self.F.append(tri)
                continue

            # ниже — старое поведение (на случай простых мешей)
            if len(tri) == 3:
                i0,i1,i2 = tri
                self.F.append(Face((i0,i1,i2),
                                   (tuple(self.UV[i0]), tuple(self.UV[i1]), tuple(self.UV[i2]))))
            elif len(tri) == 4:
                i0,i1,i2,i3 = tri
                self.F.append(Face((i0,i1,i2),
                                   (tuple(self.UV[i0]), tuple(self.UV[i1]), tuple(self.UV[i2]))))
                self.F.append(Face((i0,i2,i3),
                                   (tuple(self.UV[i0]), tuple(self.UV[i2]), tuple(self.UV[i3]))))
            else:
                base = tri[0]
                for k in range(1, len(tri)-1):
                    i0, i1, i2 = base, tri[k], tri[k+1]
                    self.F.append(Face((i0,i1,i2),
                                       (tuple(self.UV[i0]), tuple(self.UV[i1]), tuple(self.UV[i2]))))
        self.compute_face_normals()
        self.compute_vertex_normals_if_missing()

    def compute_face_normals(self):

        self.face_normals = []

        # центр модели в текущих координатах
        mesh_center = np.mean(self.V, axis=0)

        for f in self.F:
            p0, p1, p2 = self.V[list(f.idx)]
            n = normalize(np.cross(p1 - p0, p2 - p0))  # исходная нормаль (зависит от порядка вершин)

            # центр текущей грани
            face_center = (p0 + p1 + p2) / 3.0
            outward = face_center - mesh_center

            # если нормаль смотрит внутрь — переворачиваем
            if np.dot(n, outward) < 0.0:
                n = -n

            f.normal = n
            self.face_normals.append(n)

    def compute_vertex_normals_if_missing(self):
        if self.VN is not None and len(self.VN)==len(self.V):
            self.VN = np.asarray(self.VN, float)
            return
        acc = np.zeros_like(self.V)
        for f in self.F:
            n = f.normal
            for i in f.idx:
                acc[i] += n
        self.VN = np.array([normalize(v) for v in acc])

    def apply(self, M4):
        Vh = np.c_[ self.V, np.ones((len(self.V),1)) ]
        Vt = (M4 @ Vh.T).T
        Vt = Vt[:,:3] / Vt[:,[3]]
        self.V = Vt
        R = M4[:3,:3]
        self.VN = np.array([ normalize(R @ n) for n in self.VN ])
        self.compute_face_normals()
        return self

def tetrahedron_mesh():
    V = np.array([
        (1,1,1),
        (1,-1,-1),
        (-1,1,-1),
        (-1,-1,1)
    ], float)
    F = [
        (0,1,2),
        (0,3,1),
        (0,2,3),
        (1,3,2),
    ]
    UV = np.array([
        (0.5, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.5, 0.5),
    ], float)
    return Mesh(V, F, UV=UV)

def cube_mesh():
    V = np.array([
        (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
        (-1,-1, 1), (1,-1, 1), (1,1, 1), (-1,1, 1)
    ], float)
    F = [
        (0,1,2,3),
        (4,5,6,7),
        (0,1,5,4),
        (1,2,6,5),
        (2,3,7,6),
        (3,0,4,7),
    ]
    UV = np.array([
        (0,0), (1,0), (1,1), (0,1),
        (0,0), (1,0), (1,1), (0,1),
    ], float)
    return Mesh(V, F, UV=UV)

def octahedron_mesh():
    V = np.array([
        (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)
    ], float)
    F = [
        (4,0,2), (4,2,1), (4,1,3), (4,3,0),
        (5,2,0), (5,1,2), (5,3,1), (5,0,3),
    ]
    UV = np.array([
        (1,0.5), (0,0.5), (0.5,1.0), (0.5,0.0), (0.5,0.5), (0.5,0.5)
    ], float)
    return Mesh(V, F, UV=UV)

# ==========================
# OBJ загрузка (v/vt/vn)
# ==========================
def load_obj(path):
    vs, vts, vns = [], [], []
    faces_raw = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            t = line.strip().split()
            if t[0]=='v':
                vs.append(list(map(float, t[1:4])))
            elif t[0]=='vt':
                vts.append(list(map(float, t[1:3])))
            elif t[0]=='vn':
                vns.append(list(map(float, t[1:4])))
            elif t[0]=='f':
                trip = []
                for part in t[1:]:
                    parts = part.split('/')
                    vi = int(parts[0]) - 1
                    ti = int(parts[1]) - 1 if len(parts)>1 and parts[1] else None
                    ni = int(parts[2]) - 1 if len(parts)>2 and parts[2] else None
                    trip.append((vi, ti, ni))
                # триангуляция полигона
                for k in range(1, len(trip)-1):
                    faces_raw.append([ trip[0], trip[k], trip[k+1] ])

    V = np.array(vs, float) if vs else np.zeros((0,3), float)

    faces = []
    for tri in faces_raw:
        idx = tuple(vi for (vi,ti,ni) in tri)
        uv  = tuple( (tuple(vts[ti]) if (ti is not None and 0 <= ti < len(vts)) else (0.0,0.0))
                     for (vi,ti,ni) in tri )
        faces.append(Face(idx=idx, uv=uv))

    # нормали можно не тянуть из файла — вы их уже усредняете
    return Mesh(V, faces, UV=None, VN=None)


def edge_function(ax, ay, bx, by, cx, cy):
    return (cx - ax)*(by - ay) - (cy - ay)*(bx - ax)

def shade_lambert(diff_color, n, l_dir):
    #скалярное произведение нормали и направления света
    ndotl = max(0.0, float(np.dot(n, l_dir)))
    #умножаем переданный цвет на скалярное произведение
    c = np.clip(np.array(diff_color)*ndotl, 0, 1)

    return (c*255.0).astype(np.uint8)

def shade_toon(diff_color, n, l_dir, levels=4):
    ndotl = max(0.0, float(np.dot(n, l_dir)))
    
    q = int(ndotl * levels) / max(1, levels-1)
    c = np.clip(np.array(diff_color)*q, 0, 1)
    return (c*255.0).astype(np.uint8)

def sample_texture(img_np, u, v):
    h,w,_ = img_np.shape
    uu = (u % 1.0) * (w-1)
    vv = (1.0 - (v % 1.0)) * (h-1)
    x = int(uu); y = int(vv)
    return img_np[y, x]

class Renderer:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.bg = (245,245,245)
        self.light_pos = np.array([0,0,-50], float)
        self.object_color = np.array([0.9, 0.4, 0.2], float)
        self.use_texture = False
        self.texture = None
        self.shading = 'gouraud'  # 'gouraud' | 'phong-toon'
        self.perspective = True
        self.cull_backfaces = True
        self.fov = 60.0
        self.ortho_view = np.array([0.0, 0.0, 1.0], float)

    def set_texture_from_path(self, path):
        img = Image.open(path).convert('RGB')
        self.texture = np.array(img)
        self.use_texture = True

    def clear(self):
        img = Image.new('RGB', (self.w, self.h), self.bg)
        zbuf = np.full((self.h, self.w), np.inf, float)
        return img, zbuf

    def project(self, V):
        Vw = V.copy()
        Vw[:,2] += 5.0
        if self.perspective:
            f = 1.0/np.tan(np.radians(self.fov/2.0))
            x = f * Vw[:,0] / Vw[:,2]
            y = f * Vw[:,1] / Vw[:,2]
        else:
            x = Vw[:,0]; y = Vw[:,1]
        sx = (x * self.h*0.5) + self.w*0.5
        sy = (-y * self.h*0.5) + self.h*0.5
        return np.c_[sx, sy], Vw[:,2]

    def render(self, mesh: Mesh, model_matrix=None):
        work = Mesh(mesh.V.copy(), [f.idx for f in mesh.F], UV=mesh.UV.copy(), VN=mesh.VN.copy())
        if model_matrix is not None:
            work.apply(model_matrix)

        img, zbuf = self.clear()
        pix = img.load()

        screen, zview = self.project(work.V)  # work.V уже в world после model_matrix

        # направление от вершины к источнику
        Lvec = self.light_pos[None, :] - work.V
        eps = 1e-8
        Lnorm = np.linalg.norm(Lvec, axis=1, keepdims=True)
        Lnorm[Lnorm < eps] = eps
        Ldir_per_vertex = Lvec / Lnorm  # нормализованные направления света

        #растеризация треугольников
        for f in work.F:
            i0, i1, i2 = f.idx
            v0, v1, v2 = work.V[[i0, i1, i2]]

            # отсечение
            if self.cull_backfaces:
                n = f.normal
                center = (v0 + v1 + v2) / 3.0
                if self.perspective:
                    center_shifted = np.array([center[0], center[1], center[2] + 5.0], float)
                    view_to_face = normalize(center_shifted)  # вектор к центру грани
                else:
                    view_to_face = normalize(self.ortho_view)
                # лицевая грань, если нормаль смотрит К камере -> скаляр < 0
                if np.dot(n, view_to_face) >= 0.0:
                    continue

            #экранные координаты
            x0, y0 = screen[i0]
            x1, y1 = screen[i1]
            x2, y2 = screen[i2]
            z0, z1, z2 = zview[i0], zview[i1], zview[i2]

            # bbox
            minx = int(max(0, np.floor(min(x0, x1, x2))))
            maxx = int(min(self.w - 1, np.ceil(max(x0, x1, x2))))
            miny = int(max(0, np.floor(min(y0, y1, y2))))
            maxy = int(min(self.h - 1, np.ceil(max(y0, y1, y2))))

            area = edge_function(x0, y0, x1, y1, x2, y2)
            if area == 0:
                continue

            # атрибуты вершин
            n0, n1, n2 = work.VN[[i0, i1, i2]]
            l0, l1, l2 = Ldir_per_vertex[[i0, i1, i2]]
            uv0, uv1, uv2 = np.array(f.uv[0]), np.array(f.uv[1]), np.array(f.uv[2])

            #предсчитать цвета по Ламберту
            if self.shading == 'gouraud':
                c0 = shade_lambert(self.object_color, n0, l0)
                c1 = shade_lambert(self.object_color, n1, l1)
                c2 = shade_lambert(self.object_color, n2, l2)

            invz0 = 1.0 / max(1e-8, z0)
            invz1 = 1.0 / max(1e-8, z1)
            invz2 = 1.0 / max(1e-8, z2)

            # заполнение
            for y in range(miny, maxy + 1):
                for x in range(minx, maxx + 1):
                    w0 = edge_function(x1, y1, x2, y2, x, y)
                    w1 = edge_function(x2, y2, x0, y0, x, y)
                    w2 = edge_function(x0, y0, x1, y1, x, y)
                    if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                        w0 /= area
                        w1 /= area
                        w2 /= area

                        # z-buffer
                        z = w0 * z0 + w1 * z1 + w2 * z2
                        if z >= zbuf[y, x]:
                            continue
                        zbuf[y, x] = z

                        if self.shading == 'gouraud':
                            col = (w0 * c0 + w1 * c1 + w2 * c2).astype(np.uint8)
                        else:
                            # phong + toon: интерполяция нормали и света
                            n = normalize(w0 * n0 + w1 * n1 + w2 * n2)
                            l = normalize(w0 * l0 + w1 * l1 + w2 * l2)
                            col = shade_toon(self.object_color, n, l, levels=4)

                        # текстура
                        if self.use_texture and self.texture is not None:
                            den = w0 * invz0 + w1 * invz1 + w2 * invz2
                            u = (w0 * uv0[0] * invz0 + w1 * uv1[0] * invz1 + w2 * uv2[0] * invz2) / den
                            v = (w0 * uv0[1] * invz0 + w1 * uv1[1] * invz1 + w2 * uv2[1] * invz2) / den
                            uv = np.array([u, v], float)
                            tex = sample_texture(self.texture, float(uv[0]), float(uv[1])).astype(np.uint8)
                            col = (col.astype(int) * tex.astype(int) // 255).astype(np.uint8)

                        pix[x, y] = tuple(map(int, col))

        return img


# ==========================
# GUI
# ==========================
POLY = {
    "Тетраэдр": tetrahedron_mesh,
    "Куб": cube_mesh,
    "Октаэдр": octahedron_mesh,
}

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Шейдинг и текстурирование (Lambert/Gouraud, Phong-Toon)")
        self.W, self.H = 900, 600

        # === Панель 1: выбор модели / режимов ===
        top = ttk.Frame(root); top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(top, text="Модель:").pack(side=tk.LEFT)
        self.poly_var = tk.StringVar(value="Куб")
        cb = ttk.Combobox(top, textvariable=self.poly_var, values=list(POLY.keys()), state='readonly', width=12)
        cb.pack(side=tk.LEFT, padx=6)
        cb.bind('<<ComboboxSelected>>', lambda e: self.set_model())

        ttk.Label(top, text="Шейдинг:").pack(side=tk.LEFT, padx=(12,2))
        self.shading_var = tk.StringVar(value="gouraud")
        ttk.Radiobutton(top, text="Gouraud (Lambert)", variable=self.shading_var, value="gouraud", command=self.redraw).pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="Phong-Toon",        variable=self.shading_var, value="phong-toon", command=self.redraw).pack(side=tk.LEFT)

        self.cull_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Cull backfaces", variable=self.cull_var, command=self.redraw).pack(side=tk.LEFT, padx=(12,4))

        self.persp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Perspective", variable=self.persp_var, command=self.redraw).pack(side=tk.LEFT, padx=(12,4))

        self.tex_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Texture", variable=self.tex_var, command=self.toggle_texture).pack(side=tk.LEFT, padx=(12,4))

        ttk.Button(top, text="Загрузить OBJ", command=self.load_obj).pack(side=tk.RIGHT, padx=(4,0))
        ttk.Button(top, text="Загрузить текстуру", command=self.load_texture).pack(side=tk.RIGHT, padx=(4,8))

        # === Панель 2: трансформации ===
        ctrl = ttk.Frame(root); ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0,6))
        ttk.Label(ctrl, text="Поворот (deg): X").pack(side=tk.LEFT)
        self.rx = tk.DoubleVar(value=30); ttk.Entry(ctrl, textvariable=self.rx, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(ctrl, text="Y").pack(side=tk.LEFT); self.ry = tk.DoubleVar(value=30); ttk.Entry(ctrl, textvariable=self.ry, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(ctrl, text="Z").pack(side=tk.LEFT); self.rz = tk.DoubleVar(value=0);  ttk.Entry(ctrl, textvariable=self.rz, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="Смещение: X").pack(side=tk.LEFT, padx=(12,2))
        self.tx = tk.DoubleVar(value=0); ttk.Entry(ctrl, textvariable=self.tx, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(ctrl, text="Y").pack(side=tk.LEFT); self.ty = tk.DoubleVar(value=0); ttk.Entry(ctrl, textvariable=self.ty, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(ctrl, text="Z").pack(side=tk.LEFT); self.tz = tk.DoubleVar(value=0); ttk.Entry(ctrl, textvariable=self.tz, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="Масштаб:").pack(side=tk.LEFT, padx=(12,2))
        self.s = tk.DoubleVar(value=1.0); ttk.Entry(ctrl, textvariable=self.s, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="Применить трансформацию", command=self.apply_transform).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(ctrl, text="Обновить", command=self.redraw).pack(side=tk.RIGHT)

        # === Панель 3: Свет и цвет ===
        lc = ttk.Frame(root); lc.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0,6))

        ttk.Label(lc, text="Light X").pack(side=tk.LEFT)
        self.lx = tk.DoubleVar(value=3.0); ttk.Entry(lc, textvariable=self.lx, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(lc, text="Y").pack(side=tk.LEFT); self.ly = tk.DoubleVar(value=3.0); ttk.Entry(lc, textvariable=self.ly, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(lc, text="Z").pack(side=tk.LEFT); self.lz = tk.DoubleVar(value=5.0); ttk.Entry(lc, textvariable=self.lz, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Label(lc, text="   Color R").pack(side=tk.LEFT, padx=(12,2))
        self.cr = tk.DoubleVar(value=0.9); ttk.Entry(lc, textvariable=self.cr, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(lc, text="G").pack(side=tk.LEFT); self.cg = tk.DoubleVar(value=0.4); ttk.Entry(lc, textvariable=self.cg, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(lc, text="B").pack(side=tk.LEFT); self.cb = tk.DoubleVar(value=0.2); ttk.Entry(lc, textvariable=self.cb, width=5).pack(side=tk.LEFT, padx=2)

        ttk.Button(lc, text="Применить", command=self.apply_light_color).pack(side=tk.LEFT, padx=(12,4))
        ttk.Button(lc, text="Сброс", command=self.reset_light_color).pack(side=tk.LEFT)

        # === Canvas (последним, чтобы панели не прятались) ===
        self.canvas = tk.Canvas(root, width=self.W, height=self.H, bg='white', highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.renderer = Renderer(self.W, self.H)
        self.mesh = POLY[self.poly_var.get()]()
        self.tk_img = None

        self.canvas.bind("<Configure>", self.on_resize)
        self.redraw()

    def apply_transform(self):
        # собираем матрицу из текущих полей
        rx, ry, rz = float(self.rx.get()), float(self.ry.get()), float(self.rz.get())
        tx, ty, tz = float(self.tx.get()), float(self.ty.get()), float(self.tz.get())
        s = float(self.s.get())
        M = T(tx, ty, tz) @ Rz(rz) @ Ry(ry) @ Rx(rx) @ S(s, s, s)

        # ЗАПЕКАЕМ в сам меш (инкрементально)
        self.mesh.apply(M)

        # Сбрасываем поля, чтобы следующий «Применить» был новым инкрементом
        self.rx.set(0.0);
        self.ry.set(0.0);
        self.rz.set(0.0)
        self.tx.set(0.0);
        self.ty.set(0.0);
        self.tz.set(0.0)
        self.s.set(1.0)

        # Перерисовка
        self.redraw()

    def set_model(self):
        self.mesh = POLY[self.poly_var.get()]()
        self.redraw()

    def load_obj(self):
        p = filedialog.askopenfilename(title="Открыть OBJ", filetypes=[("OBJ", "*.obj"), ("All","*.*")])
        if not p: return
        try:
            self.mesh = load_obj(p)
            messagebox.showinfo("OK", "OBJ загружен.")
            self.redraw()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить OBJ:\n{e}")

    def load_texture(self):
        p = filedialog.askopenfilename(title="Открыть текстуру", filetypes=[("Изображения","*.png;*.jpg;*.jpeg;*.bmp"),("All","*.*")])
        if not p: return
        self.renderer.set_texture_from_path(p)
        self.tex_var.set(True)
        self.redraw()

    def toggle_texture(self):
        self.renderer.use_texture = self.tex_var.get()
        self.redraw()

    def on_resize(self, ev):
        self.renderer.w, self.renderer.h = ev.width, ev.height
        self.W, self.H = ev.width, ev.height
        self.redraw()

    def apply_light_color(self):
        # Свет
        lx, ly, lz = self.lx.get(), self.ly.get(), self.lz.get()
        self.renderer.light_pos = np.array([lx, ly, lz], float)
        # Цвет (clamp 0..1)
        cr = max(0.0, min(1.0, self.cr.get()))
        cg = max(0.0, min(1.0, self.cg.get()))
        cb = max(0.0, min(1.0, self.cb.get()))
        self.renderer.object_color = np.array([cr, cg, cb], float)
        self.redraw()

    def reset_light_color(self):
        self.lx.set(3.0); self.ly.set(3.0); self.lz.set(5.0)
        self.cr.set(0.9); self.cg.set(0.4); self.cb.set(0.2)
        self.apply_light_color()

    def redraw(self):
        self.renderer.shading = self.shading_var.get()
        self.renderer.perspective = self.persp_var.get()
        self.renderer.cull_backfaces = self.cull_var.get()

        # текущая матрица ТОЛЬКО для предпросмотра (не мутируем меш)
        rx, ry, rz = float(self.rx.get()), float(self.ry.get()), float(self.rz.get())
        tx, ty, tz = float(self.tx.get()), float(self.ty.get()), float(self.tz.get())
        s = float(self.s.get())
        M_preview = T(tx, ty, tz) @ Rz(rz) @ Ry(ry) @ Rx(rx) @ S(s, s, s)

        img = self.renderer.render(self.mesh, model_matrix=M_preview)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
