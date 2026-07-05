import argparse
import math

import moderngl
import numpy as np
import pygame


# ---------- Display & World ----------
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1080, 720
WORLD_SIZE = 192.0
WORLD_X = WORLD_Y = WORLD_Z = WORLD_SIZE


# ---------- Original 2D Dynamics, Lifted To 3D ----------
NUM_PARTICLES = 12_000
ATTRACTION_K = 64.0
RATIO_INIT = 0.25
FRICTION_INIT = 0.5
CENTER_PULL_INIT = 4.0
MAX_RADIUS = 32.0
BIN_SIZE = 16.0
COMPUTE_GROUP_SIZE = 256
GRID_X = int(math.ceil(WORLD_X / BIN_SIZE))
GRID_Y = int(math.ceil(WORLD_Y / BIN_SIZE))
GRID_Z = int(math.ceil(WORLD_Z / BIN_SIZE))
NUM_BINS = GRID_X * GRID_Y * GRID_Z
MAX_BIN_PARTICLES = 256
SEARCH_RANGE = int((MAX_RADIUS + BIN_SIZE - 1) // BIN_SIZE)
GENOME_DRIFT_INIT = 1
DIVERGENCE_INIT = 4
CENTER_COHESION_FORCE = 2.0
DENSITY_TARGET_GRID = 1
DENSITY_TARGET_PERCENTILE = 75.0
DENSITY_TARGET_UPDATE_INTERVAL = 15
DENSITY_TARGET_SMOOTHING = 0.18
INTERACTION_VISIBILITY_LOW = 10.0
INTERACTION_VISIBILITY_HIGH = 25.0


class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.label = label
        self.dragging = False
        self.handle_radius = height // 2

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            handle_x = self.rect.x + (self.val - self.min_val) / (self.max_val - self.min_val) * self.rect.width
            handle_rect = pygame.Rect(
                handle_x - self.handle_radius,
                self.rect.y,
                self.handle_radius * 2,
                self.rect.height,
            )
            if handle_rect.collidepoint(mouse_pos):
                self.dragging = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_x = pygame.mouse.get_pos()[0]
            relative_x = max(0, min(self.rect.width, mouse_x - self.rect.x))
            self.val = self.min_val + (relative_x / self.rect.width) * (self.max_val - self.min_val)
            return True
        return False

    def draw(self, surface, font):
        pygame.draw.rect(surface, (100, 100, 100), self.rect)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2)
        handle_x = int(self.rect.x + (self.val - self.min_val) / (self.max_val - self.min_val) * self.rect.width)
        pygame.draw.circle(surface, (255, 255, 255), (handle_x, self.rect.centery), self.handle_radius)
        pygame.draw.circle(surface, (150, 150, 150), (handle_x, self.rect.centery), self.handle_radius, 2)
        text = font.render(f"{self.label}: {self.val:.3f}", True, (255, 255, 255))
        surface.blit(text, (self.rect.x, self.rect.y - 25))


def wrap_delta_np(delta, world):
    return delta - world * np.round(delta / world)


def circular_mean(values, world_size, fallback):
    if values.size == 0:
        return fallback
    angles = values * (2.0 * math.pi / world_size)
    s = float(np.mean(np.sin(angles)))
    c = float(np.mean(np.cos(angles)))
    if math.hypot(s, c) < 1e-4:
        return fallback
    angle = math.atan2(s, c)
    if angle < 0.0:
        angle += 2.0 * math.pi
    return angle * (world_size / (2.0 * math.pi))


def compute_density_target(position_data, previous_target):
    coords = np.asarray(position_data[:, :3], dtype=np.float32)
    if coords.size == 0:
        return previous_target.copy()

    grid = DENSITY_TARGET_GRID
    scaled = np.floor(coords / np.array([WORLD_X, WORLD_Y, WORLD_Z], dtype=np.float32) * grid).astype(np.int32)
    scaled = np.clip(scaled, 0, grid - 1)
    bin_ids = scaled[:, 0] + scaled[:, 1] * grid + scaled[:, 2] * grid * grid
    counts = np.bincount(bin_ids, minlength=grid * grid * grid)
    occupied = counts[counts > 0]
    if occupied.size == 0:
        return previous_target.copy()

    threshold = max(2.0, float(np.percentile(occupied, DENSITY_TARGET_PERCENTILE)))
    dense_mask = counts[bin_ids] >= threshold
    dense_coords = coords[dense_mask]
    if dense_coords.shape[0] < max(64, NUM_PARTICLES // 64):
        dense_coords = coords

    target = np.array([
        circular_mean(dense_coords[:, 0], WORLD_X, float(previous_target[0])),
        circular_mean(dense_coords[:, 1], WORLD_Y, float(previous_target[1])),
        circular_mean(dense_coords[:, 2], WORLD_Z, float(previous_target[2])),
    ], dtype=np.float32)
    return target


def perspective(fovy_radians, aspect, near, far):
    f = 1.0 / math.tan(fovy_radians / 2.0)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2.0 * far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat


def look_at(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    mat = np.identity(4, dtype=np.float32)
    mat[0, :3] = s
    mat[1, :3] = u
    mat[2, :3] = -f
    mat[0, 3] = -np.dot(s, eye)
    mat[1, 3] = -np.dot(u, eye)
    mat[2, 3] = np.dot(f, eye)
    return mat


class FreeCamera:
    def __init__(self, position, yaw=0.0, pitch=0.0):
        self.position = np.array(position, dtype=np.float32)
        self.yaw = yaw
        self.pitch = pitch
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def forward(self):
        cos_p = math.cos(self.pitch)
        return np.array([
            cos_p * math.cos(self.yaw),
            math.sin(self.pitch),
            cos_p * math.sin(self.yaw),
        ], dtype=np.float32)

    def right(self):
        fwd = self.forward()
        r = np.cross(fwd, self.world_up)
        r /= np.linalg.norm(r)
        return r

    def up(self):
        r = self.right()
        fwd = self.forward()
        u = np.cross(r, fwd)
        u /= np.linalg.norm(u)
        return u

    def view_matrix(self):
        return look_at(self.position, self.position + self.forward(), self.world_up)

    def process_mouse(self, dx, dy, sensitivity=0.002):
        self.yaw += dx * sensitivity
        self.pitch -= dy * sensitivity
        self.pitch = max(-math.pi / 2.01, min(math.pi / 2.01, self.pitch))

    def move(self, direction, amount):
        self.position += direction * amount


def main(batch_frames=0, seed=None):
    batch_mode = batch_frames > 0
    if seed is not None:
        np.random.seed(seed)

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_caption("Genetic Particle Life - Faithful 2D to 3D")
    display_flags = pygame.OPENGL | pygame.DOUBLEBUF
    if batch_mode:
        display_flags |= pygame.HIDDEN
    pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), display_flags)
    pygame.mouse.set_visible(batch_mode)
    pygame.event.set_grab(False if batch_mode else True)
    mouse_grabbed = False if batch_mode else True

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    ui_surface = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.SRCALPHA)
    font = pygame.font.Font(None, 24)

    ratio_slider = Slider(50, 50, 260, 20, 0.1, 1.00, RATIO_INIT, "Repulsion/Attraction Ratio")
    friction_slider = Slider(50, 100, 260, 20, 0.05, 0.99, FRICTION_INIT, "Particle Drift Strength")
    center_pull_slider = Slider(50, 150, 260, 20, 0.0, 16.0, CENTER_PULL_INIT, "Density Center Pull")
    genome_drift_slider = Slider(50, 200, 260, 20, 0.0, 8.0, GENOME_DRIFT_INIT, "Genome Drift")
    divergence_slider = Slider(50, 250, 260, 20, 0.0, 8.0, DIVERGENCE_INIT, "Divergence Strength")
    sliders = [ratio_slider, friction_slider, center_pull_slider, genome_drift_slider, divergence_slider]

    positions = np.ones((NUM_PARTICLES, 4), dtype=np.float32)
    positions[:, 0] = np.random.rand(NUM_PARTICLES).astype(np.float32) * WORLD_X
    positions[:, 1] = np.random.rand(NUM_PARTICLES).astype(np.float32) * WORLD_Y
    positions[:, 2] = np.random.rand(NUM_PARTICLES).astype(np.float32) * WORLD_Z
    cohesion_target = compute_density_target(
        positions,
        np.array([WORLD_X * 0.5, WORLD_Y * 0.5, WORLD_Z * 0.5], dtype=np.float32),
    )

    velocities = np.zeros((NUM_PARTICLES, 4), dtype=np.float32)
    velocities[:, :3] = np.random.uniform(-8, 8, (NUM_PARTICLES, 3)).astype(np.float32)

    raw_genomes = np.random.uniform(-1, 1, (NUM_PARTICLES, 6)).astype(np.float32)
    genomes = raw_genomes / np.linalg.norm(raw_genomes, axis=1, keepdims=True)

    pos_buf = ctx.buffer(positions.tobytes(), dynamic=True)
    vel_buf = ctx.buffer(velocities.tobytes(), dynamic=True)
    genome_buf = ctx.buffer(genomes.tobytes(), dynamic=True)
    bin_counts_buf = ctx.buffer(reserve=NUM_BINS * 4, dynamic=True)
    bin_particles_buf = ctx.buffer(reserve=NUM_BINS * MAX_BIN_PARTICLES * 4, dynamic=True)

    vertex_shader = """
    #version 430
    in vec4 in_pos;
    in vec4 in_genome;
    out vec4 v_color;
    out float v_size_fade;
    out float v_visibility;

    uniform mat4 mvp;
    uniform float point_scale;
    uniform float orb_size;

    void main() {
        vec4 clip = mvp * vec4(in_pos.xyz, 1.0);
        gl_Position = clip;
        float inv_w = max(0.0001, 1.0 / clip.w);
        gl_PointSize = max(2.0, orb_size * point_scale * inv_w);
        v_visibility = clamp(in_pos.w, 0.0, 1.0);
        v_color = vec4(in_genome.xyz * 0.5 + 0.5, v_visibility);
        v_size_fade = clamp(inv_w * 520.0, 0.20, 1.0);
    }
    """

    fragment_shader = """
    #version 430
    in vec4 v_color;
    in float v_size_fade;
    in float v_visibility;
    out vec4 fragColor;

    vec3 saturate_color(vec3 c, float amount) {
        float luma = dot(c, vec3(0.299, 0.587, 0.114));
        return clamp(mix(vec3(luma), c, amount), 0.0, 1.0);
    }

    void main() {
        vec2 uv = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(uv, uv);
        if (r2 > 1.0) discard;

        float z = sqrt(max(0.0, 1.0 - r2));
        vec3 normal = normalize(vec3(uv.x, -uv.y, z));
        vec3 light_dir = normalize(vec3(-0.38, 0.55, 0.74));

        float diffuse = max(dot(normal, light_dir), 0.0);
        float center = exp(-2.45 * r2);
        float edge = smoothstep(1.0, 0.72, sqrt(r2));
        float rim = pow(1.0 - z, 2.3);
        float spec = pow(max(dot(reflect(-light_dir, normal), vec3(0.0, 0.0, 1.0)), 0.0), 24.0);

        vec3 base = saturate_color(v_color.rgb, 1.45);
        vec3 color = base * (0.30 + 0.76 * diffuse + 0.46 * center);
        color += base * rim * 0.16;
        color += vec3(1.0) * spec * 0.18;
        color *= mix(0.76, 1.08, v_size_fade);

        float alpha = clamp((0.12 + 0.88 * center) * edge * v_visibility, 0.0, 1.0);
        if (alpha < 0.004) discard;
        fragColor = vec4(clamp(color, 0.0, 1.0), alpha);
    }
    """

    prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    vao = ctx.vertex_array(prog, [
        (pos_buf, "4f", "in_pos"),
        (genome_buf, "4f 8x", "in_genome"),
    ])

    binning_shader = ctx.compute_shader(f"""
    #version 430
    layout(local_size_x = {COMPUTE_GROUP_SIZE}) in;

    layout(std430, binding=0) buffer Positions {{ vec4 pos[]; }};
    layout(std430, binding=6) buffer BinCounts {{ uint bin_counts[]; }};
    layout(std430, binding=7) buffer BinParticles {{ uint bin_particles[]; }};

    uniform float bin_size;
    uniform int grid_x;
    uniform int grid_y;
    uniform int grid_z;
    uniform int num_particles;

    uint bin_index(ivec3 cell) {{
        return uint(cell.x + cell.y * grid_x + cell.z * grid_x * grid_y);
    }}

    void main() {{
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(num_particles)) return;

        vec3 p = pos[i].xyz;
        ivec3 cell = ivec3(
            clamp(int(p.x / bin_size), 0, grid_x - 1),
            clamp(int(p.y / bin_size), 0, grid_y - 1),
            clamp(int(p.z / bin_size), 0, grid_z - 1)
        );

        uint b = bin_index(cell);
        uint offset = atomicAdd(bin_counts[b], 1u);
        if (offset < {MAX_BIN_PARTICLES}u) {{
            bin_particles[b * {MAX_BIN_PARTICLES}u + offset] = i;
        }}
    }}
    """)

    interaction_shader = ctx.compute_shader(f"""
    #version 430
    layout(local_size_x = {COMPUTE_GROUP_SIZE}) in;

    layout(std430, binding=0) buffer Positions {{ vec4 pos[]; }};
    layout(std430, binding=1) buffer Velocities {{ vec4 vel[]; }};
    layout(std430, binding=2) buffer Genomes {{ float genomes[]; }};
    layout(std430, binding=6) readonly buffer BinCounts {{ uint bin_counts[]; }};
    layout(std430, binding=7) readonly buffer BinParticles {{ uint bin_particles[]; }};

    uniform int num_particles;
    uniform float world_x;
    uniform float world_y;
    uniform float world_z;
    uniform float K_attraction;
    uniform float K_repulsion;
    uniform float friction;
    uniform float delta_time;
    uniform float max_radius;
    uniform float bin_size;
    uniform int grid_x;
    uniform int grid_y;
    uniform int grid_z;
    uniform vec3 cohesion_target;
    uniform float center_cohesion_strength;
    uniform float genome_drift_speed;
    uniform float divergence_strength;

    int wrap(int value, int max_value) {{
        return (value % max_value + max_value) % max_value;
    }}

    uint bin_index(ivec3 cell) {{
        return uint(cell.x + cell.y * grid_x + cell.z * grid_x * grid_y);
    }}

    vec3 wrap_delta(vec3 d) {{
        vec3 half_world = vec3(world_x, world_y, world_z) * 0.5;
        if (d.x > half_world.x) d.x -= world_x;
        else if (d.x < -half_world.x) d.x += world_x;
        if (d.y > half_world.y) d.y -= world_y;
        else if (d.y < -half_world.y) d.y += world_y;
        if (d.z > half_world.z) d.z -= world_z;
        else if (d.z < -half_world.z) d.z += world_z;
        return d;
    }}

    void main() {{
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(num_particles)) return;

        vec3 p = pos[i].xyz;
        vec3 v = vel[i].xyz;

        float my_genome[6];
        for (int g = 0; g < 6; g++) {{
            my_genome[g] = genomes[i * 6 + g];
        }}

        float genetic_repulsion[6] = float[6](0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        float genetic_weight = 0.0;
        float interaction_count = 0.0;

        vec3 f = vec3(0.0);
        ivec3 center = ivec3(
            clamp(int(p.x / bin_size), 0, grid_x - 1),
            clamp(int(p.y / bin_size), 0, grid_y - 1),
            clamp(int(p.z / bin_size), 0, grid_z - 1)
        );

        const int sr = {SEARCH_RANGE};
        for (int dx = -sr; dx <= sr; dx++) {{
            int nx = wrap(center.x + dx, grid_x);
            for (int dy = -sr; dy <= sr; dy++) {{
                int ny = wrap(center.y + dy, grid_y);
                for (int dz = -sr; dz <= sr; dz++) {{
                    int nz = wrap(center.z + dz, grid_z);
                    uint bin_idx = bin_index(ivec3(nx, ny, nz));
                    uint count = min(bin_counts[bin_idx], {MAX_BIN_PARTICLES}u);

                    for (uint b = 0u; b < count; b++) {{
                        uint j = bin_particles[bin_idx * {MAX_BIN_PARTICLES}u + b];
                        if (j == i) continue;

                        vec3 d = wrap_delta(pos[j].xyz - p);
                        float dist = length(d);
                        if (dist > max_radius || dist < 0.1) continue;
                        vec3 dn = d / dist;
                        interaction_count += 1.0;

                        float other_genome[6];
                        float compatibility = 0.0;
                        for (int g = 0; g < 6; g++) {{
                            other_genome[g] = genomes[j * 6 + g];
                            compatibility += my_genome[g] * other_genome[g];
                        }}

                        float mind = 8.0;

                        if (dist < mind) {{
                            f -= dn * 8.0 * (1.0 - dist / mind) * K_repulsion;
                        }} else if (dist < max_radius) {{
                            f += dn * compatibility * (1.0 - dist / max_radius) * K_attraction;

                            float similarity = compatibility;
                            float threshold = 0.5;
                            if (similarity > threshold) {{
                                float force = (similarity - threshold) * divergence_strength * (1.0 - dist / max_radius);
                                for (int g = 0; g < 6; g++) {{
                                    genetic_repulsion[g] += (my_genome[g] - other_genome[g]) * force;
                                }}
                                genetic_weight += force;
                            }}
                        }}
                    }}
                }}
            }}
        }}

        if (center_cohesion_strength > 0.0001) {{
            vec3 to_center = wrap_delta(cohesion_target - p);
            float center_dist = length(to_center);
            float far_gate = smoothstep(max_radius * 0.85, min(min(world_x, world_y), world_z) * 0.42, center_dist);
            if (center_dist > 0.001 && far_gate > 0.001) {{
                f += (to_center / center_dist) * far_gate * center_cohesion_strength * {CENTER_COHESION_FORCE:.8f};
            }}
        }}

        float max_force = 2000.0;
        if (length(f) > max_force) f = normalize(f) * max_force;

        v += f * delta_time;
        v *= friction;

        float max_vel = 150.0;
        if (length(v) > max_vel) v = normalize(v) * max_vel;

        p += v * delta_time;

        if (p.x < 0.0) p.x += world_x;
        else if (p.x >= world_x) p.x -= world_x;
        if (p.y < 0.0) p.y += world_y;
        else if (p.y >= world_y) p.y -= world_y;
        if (p.z < 0.0) p.z += world_z;
        else if (p.z >= world_z) p.z -= world_z;

        float visibility = smoothstep(
            {INTERACTION_VISIBILITY_LOW:.8f},
            {INTERACTION_VISIBILITY_HIGH:.8f},
            sqrt(interaction_count)
        );
        visibility = pow(visibility, 1.15);

        pos[i] = vec4(p, visibility);
        vel[i] = vec4(v, 0.0);

        float len_sq = 0.0;
        float genetic_density_scale = genetic_weight > 1.0 ? 1.0 / genetic_weight : 1.0;
        for (int g = 0; g < 6; g++) {{
            my_genome[g] += genetic_repulsion[g] * genetic_density_scale * genome_drift_speed * delta_time;
            len_sq += my_genome[g] * my_genome[g];
        }}

        float inv_len = inversesqrt(max(len_sq, 0.000001));
        for (int g = 0; g < 6; g++) {{
            genomes[i * 6 + g] = my_genome[g] * inv_len;
        }}
    }}
    """)

    clear_counts_shader = ctx.compute_shader("""
    #version 430
    layout(local_size_x = 256) in;
    layout(std430, binding=6) buffer BinCounts { uint bin_counts[]; };
    uniform int num_bins;
    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i < uint(num_bins)) bin_counts[i] = 0u;
    }
    """)

    for idx, buf in enumerate([pos_buf, vel_buf, genome_buf, None, None, None, bin_counts_buf, bin_particles_buf]):
        if buf is not None:
            buf.bind_to_storage_buffer(idx)

    def update_forces():
        attraction = ATTRACTION_K
        interaction_shader["K_attraction"].value = attraction
        interaction_shader["K_repulsion"].value = attraction * ratio_slider.val
        interaction_shader["friction"].value = friction_slider.val
        interaction_shader["center_cohesion_strength"].value = center_pull_slider.val
        interaction_shader["genome_drift_speed"].value = genome_drift_slider.val
        interaction_shader["divergence_strength"].value = divergence_slider.val

    update_forces()

    for shader, name, value in [
        (interaction_shader, "num_particles", NUM_PARTICLES),
        (interaction_shader, "world_x", float(WORLD_X)),
        (interaction_shader, "world_y", float(WORLD_Y)),
        (interaction_shader, "world_z", float(WORLD_Z)),
        (interaction_shader, "max_radius", float(MAX_RADIUS)),
        (interaction_shader, "bin_size", float(BIN_SIZE)),
        (interaction_shader, "grid_x", GRID_X),
        (interaction_shader, "grid_y", GRID_Y),
        (interaction_shader, "grid_z", GRID_Z),
        (binning_shader, "num_particles", NUM_PARTICLES),
        (binning_shader, "bin_size", float(BIN_SIZE)),
        (binning_shader, "grid_x", GRID_X),
        (binning_shader, "grid_y", GRID_Y),
        (binning_shader, "grid_z", GRID_Z),
        (clear_counts_shader, "num_bins", NUM_BINS),
    ]:
        shader[name].value = value
    interaction_shader["cohesion_target"].value = tuple(float(v) for v in cohesion_target)

    camera = FreeCamera(
        position=[WORLD_X / 2.0, WORLD_Y / 2.0, -WORLD_SIZE * 1.30],
        yaw=math.pi / 2.0,
        pitch=0.0,
    )
    proj = perspective(math.radians(55.0), DISPLAY_WIDTH / DISPLAY_HEIGHT, 0.1, 4000.0)
    prog["point_scale"].value = DISPLAY_HEIGHT / (2.0 * math.tan(math.radians(55.0) / 2.0))
    prog["orb_size"].value = 2

    ui_tex = ctx.texture((DISPLAY_WIDTH, DISPLAY_HEIGHT), 4, dtype="f1")
    ui_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    ui_vs = """
    #version 430
    in vec2 in_position;
    in vec2 in_texcoord;
    out vec2 v_texcoord;
    void main() {
        gl_Position = vec4(in_position, 0.0, 1.0);
        v_texcoord = in_texcoord;
    }
    """
    ui_fs = """
    #version 430
    uniform sampler2D ui_texture;
    in vec2 v_texcoord;
    out vec4 fragColor;
    void main() {
        fragColor = texture(ui_texture, v_texcoord);
    }
    """
    ui_prog = ctx.program(vertex_shader=ui_vs, fragment_shader=ui_fs)
    quad = np.array([
        -1.0, -1.0, 0.0, 1.0,
         1.0, -1.0, 1.0, 1.0,
        -1.0,  1.0, 0.0, 0.0,
         1.0,  1.0, 1.0, 0.0,
    ], dtype=np.float32)
    ui_vbo = ctx.buffer(quad.tobytes())
    ui_vao = ctx.vertex_array(ui_prog, [(ui_vbo, "2f 2f", "in_position", "in_texcoord")])

    clock = pygame.time.Clock()
    running = True
    paused = False
    frame_count = 0
    mouse_sensitivity = 0.002
    move_speed = 180.0

    while running:
        dt = (1.0 / 60.0) if batch_mode else min(clock.tick(60) / 1000.0, 1.0 / 20.0)
        step_once = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_TAB:
                    mouse_grabbed = not mouse_grabbed
                    pygame.mouse.set_visible(not mouse_grabbed)
                    pygame.event.set_grab(mouse_grabbed)
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_n and paused:
                    step_once = True
            elif event.type == pygame.MOUSEMOTION and mouse_grabbed:
                camera.process_mouse(event.rel[0], event.rel[1], mouse_sensitivity)

            if not mouse_grabbed:
                for slider in sliders:
                    if slider.handle_event(event):
                        update_forces()

        keys = pygame.key.get_pressed()
        fwd = camera.forward()
        right = camera.right()
        up = camera.up()
        move_dir = np.zeros(3, dtype=np.float32)
        if keys[pygame.K_w]:
            move_dir += fwd
        if keys[pygame.K_s]:
            move_dir -= fwd
        if keys[pygame.K_a]:
            move_dir -= right
        if keys[pygame.K_d]:
            move_dir += right
        if keys[pygame.K_SPACE]:
            move_dir += up
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            move_dir -= up
        if np.linalg.norm(move_dir) > 0.0:
            move_dir /= np.linalg.norm(move_dir)
            camera.move(move_dir, move_speed * dt)

        ctx.clear(0.0, 0.0, 0.0, 1.0)

        if not paused or step_once:
            if frame_count % DENSITY_TARGET_UPDATE_INTERVAL == 0:
                positions_snapshot = np.frombuffer(pos_buf.read(), dtype=np.float32).reshape(-1, 4)
                target_now = compute_density_target(positions_snapshot, cohesion_target)
                world_vec = np.array([WORLD_X, WORLD_Y, WORLD_Z], dtype=np.float32)
                target_delta = wrap_delta_np(target_now - cohesion_target, world_vec)
                cohesion_target = (cohesion_target + target_delta * DENSITY_TARGET_SMOOTHING) % world_vec
                interaction_shader["cohesion_target"].value = tuple(float(v) for v in cohesion_target)

            clear_counts_shader.run(group_x=(NUM_BINS + 255) // 256)
            binning_shader.run(group_x=(NUM_PARTICLES + COMPUTE_GROUP_SIZE - 1) // COMPUTE_GROUP_SIZE)
            ctx.memory_barrier(barriers=moderngl.SHADER_STORAGE_BARRIER_BIT)

            interaction_shader["delta_time"].value = dt
            interaction_shader.run(group_x=(NUM_PARTICLES + COMPUTE_GROUP_SIZE - 1) // COMPUTE_GROUP_SIZE)
            ctx.memory_barrier(barriers=moderngl.SHADER_STORAGE_BARRIER_BIT)

        view = camera.view_matrix()
        mvp = proj @ view
        prog["mvp"].write(mvp.T.astype("f4").tobytes())
        vao.render(moderngl.POINTS)

        ui_surface.fill((0, 0, 0, 0))
        if not batch_mode:
            for slider in sliders:
                slider.draw(ui_surface, font)
            info_lines = [
                "Faithful 2D rules lifted to 3D",
                "WASD/mouse | SPACE/SHIFT vertical | TAB sliders | P pause | N step | Q quit",
                f"particles {NUM_PARTICLES} | radius {MAX_RADIUS:.0f} | density-center pull | genome sliders",
            ]
            for idx, line in enumerate(info_lines):
                ui_surface.blit(font.render(line, True, (220, 230, 255)), (50, DISPLAY_HEIGHT - 90 + idx * 22))

        ui_tex.write(pygame.image.tostring(ui_surface, "RGBA"))
        ui_tex.use(0)
        ui_prog["ui_texture"].value = 0
        ui_vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()
        frame_count += 1
        if batch_mode and frame_count >= batch_frames:
            running = False

    pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(description="Faithful 3D port of the supplied 2D Genetic Particle Life.")
    parser.add_argument("--batch", action="store_true", help="Run hidden for a fixed number of frames, then exit.")
    parser.add_argument("--frames", type=int, default=120, help="Frames to run with --batch.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(batch_frames=args.frames if args.batch else 0, seed=args.seed)
