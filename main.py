from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0.0, exposure=0.8)
scene.set_floor(-1, (1.0, 1.0, 1.0))
scene.set_background_color((0.3, 0.4, 0.6))
scene.set_directional_light((-1, 1, 0.3), 0.0, (1, 1, 1))

BLACK = vec3(0.0, 0.0, 0.0)
WHITE = vec3(1.0, 1.0, 1.0)
SILVER = vec3(0.752941, 0.752941, 0.752941)
YELLOW = vec3(0.882352, 0.694117, 0.278431)
ORANGE = vec3(0.882352, 0.584313, 0.137254)
SKIN = vec3(0.917647, 0.701960, 0.541176)

@ti.func
def bound(x, y, z):
    return 2 * int(max(z,max(x, y)))
@ti.func
def project(n, t, p):
    y = dot(p,n);xz=p-(n*y);bt=cross(t,n);return vec3(dot(xz,t), y, dot(xz, bt))
@ti.func
def cube(x, y, z, p):
    q=ti.abs(p)-vec3(x,y,z)
    return ti.max(q, 0.0).norm() + ti.min(ti.max(q.x, ti.max(q.y, q.z)), 0.0) < 0
@ti.func
def cylinder(r1, h, r2, p):
    ms=min(r1,min(h,r2));r=vec2(p.x/r1,p.z/r2);d=vec2((r.norm()-1.0)*ms,ti.abs(p.y)-h)
    return min(max(d.x,d.y),0.0)+max(d,0.0).norm()<0
@ti.func
def sphere(rx, ry, rz, p):
    r = p/vec3(rx,ry,rz); return ti.sqrt(dot(r,r))<1
@ti.func
def meta(func: ti.template(), x, y, z, pos, dir, up, color):
    max_r = bound(x,y,z);dir = normalize(dir);up = normalize(cross(cross(dir, up), dir))
    for i,j,k in ti.ndrange((-max_r,max_r),(-max_r,max_r),(-max_r,max_r)): 
        xyz = project(dir, up, vec3(i,j,k))
        if func(x, y, z, xyz):
            scene.set_voxel(pos + vec3(i,j,k), 1, color)

@ti.kernel
def initialize_voxels():
    # body
    meta(cube,5.0,15.0,8.0,vec3(9,-24,0),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),ORANGE)
    meta(cube,2.0,25.0,1.0,vec3(9,-34,-8),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,2.0,25.0,1.0,vec3(9,-34,8),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    # legs
    meta(cylinder,6.0,2.0,3.0,vec3(6,-61,-5),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),YELLOW)
    meta(cube,4.0,10.0,3.0,vec3(9,-49,-5),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),ORANGE)
    meta(cylinder,6.0,2.0,3.0,vec3(6,-61,5),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),YELLOW)
    meta(cube,4.0,10.0,3.0,vec3(9,-49,5),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),ORANGE)
    meta(cube,2.0,10.0,1.0,vec3(9,-49,2),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,2.0,10.0,1.0,vec3(9,-49,-2),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    # head
    meta(cube,8.0,10.0,10.0,vec3(9,1,0),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),SKIN)
    meta(cube,10.0,2.0,12.0,vec3(9,12,0),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,1.5,12.0,12.0,vec3(18,2,0),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,1.0,3.0,12.0,vec3(1,11,0),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,1.0,1.5,1.5,vec3(1,5,-6),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),WHITE)
    meta(cube,1.0,1.5,1.5,vec3(1,5,-4),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,1.0,1.5,1.5,vec3(1,5,6),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),WHITE)
    meta(cube,1.0,1.5,1.5,vec3(1,5,4),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,1.0,1.5,2.5,vec3(1,-2,0),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),WHITE)
    meta(cube,1.0,1.5,2.5,vec3(1,-5,0),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,10.0,3.5,1.0,vec3(9,10,11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,10.0,3.5,1.0,vec3(9,10,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,7.5,2.0,1.0,vec3(11,5,11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,7.5,2.0,1.0,vec3(11,5,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,6.0,2.0,1.0,vec3(13,2,11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,6.0,2.0,1.0,vec3(13,2,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,4.5,2.0,1.0,vec3(14,-1,11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,4.5,2.0,1.0,vec3(14,-1,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,3.0,2.0,1.0,vec3(16,-4,11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    meta(cube,3.0,2.0,1.0,vec3(16,-4,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),BLACK)
    # arms
    meta(cube,10.0,4.0,3.0,vec3(4,-13,11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,-0.0),ORANGE)
    meta(cube,10.0,2.0,1.0,vec3(4,-13,14),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,-0.0),BLACK)
    meta(cylinder,3.0,6.0,3.0,vec3(-10,-13,11),vec3(-1.0,-0.0,0.0),vec3(-0.0,1.0,0.0),SKIN)
    meta(cube,4.0,10.0,3.0,vec3(9,-19,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,-0.0),ORANGE)
    meta(cube,2.0,10.0,1.0,vec3(9,-19,-14),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,-0.0),BLACK)
    meta(cube,2.5,2.5,2.5,vec3(-16,-13,11),vec3(0.0,1.0,0.0),vec3(1.0,0.0,0.0),SKIN)
    meta(cylinder,3.0,6.0,3.0,vec3(1,-26,-11),vec3(-1.0,-0.0,0.0),vec3(-0.0,1.0,0.0),SKIN)
    meta(cube,2.5,2.5,2.5,vec3(-5,-26,-11),vec3(0.0,1.0,0.0),vec3(1.0,0.0,0.0),SKIN)
    # nunchucks
    meta(cylinder,1.5,10.0,1.5,vec3(-5,-24,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),YELLOW)
    meta(sphere,1.0,1.0,1.0,vec3(-5,-13,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),SILVER)
    meta(sphere,1.0,1.0,1.0,vec3(-5,-11,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),SILVER)
    meta(sphere,1.0,1.0,1.0,vec3(-5,-9,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),SILVER)
    meta(sphere,1.0,1.0,1.0,vec3(-5,-9,-13),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),SILVER)
    meta(sphere,1.0,1.0,1.0,vec3(-4,-10,-14),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),SILVER)
    meta(sphere,1.0,1.0,1.0,vec3(-2,-10,-14),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),SILVER)
    meta(sphere,1.0,1.0,1.0,vec3(-2,-10,-16),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),SILVER)
    meta(sphere,1.0,1.0,1.0,vec3(-1,-11,-17),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),SILVER)
    meta(cylinder,1.5,10.0,1.5,vec3(-1,-22,-17),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),YELLOW)
initialize_voxels()
scene.finish()
