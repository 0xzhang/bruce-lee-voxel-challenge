from scene import Scene
import taichi as ti
from taichi.math import *
scene = Scene(voxel_edges=0.0, exposure=0.8)
scene.set_floor(-1, (1.0, 1.0, 1.0))
scene.set_background_color((0.3, 0.4, 0.6))
scene.set_directional_light((-1, 1, 0.3), 0.0, (1, 1, 1))
@ti.func
def rgb(r,g,b):
    return vec3(r/255.0, g/255.0, b/255.0)
@ti.func
def proj_plane(o, n, t, p): 
    y = dot(p-o,n);xz=p-(o+n*y);bt=cross(t,n);return vec3(dot(xz,t), y, dot(xz, bt))
@ti.func
def elli(rx,ry,rz,p1_unused,p2_unused,p3_unused,p):
    r = p/vec3(rx,ry,rz); return ti.sqrt(dot(r,r))<1
@ti.func
def cyli(r1,h,r2,round, cone, hole_unused, p):
    ms=min(r1,min(h,r2));rr=ms*round;rt=mix(cone*(max(ms-rr,0)),0,float(h-p.y)*0.5/h);r=vec2(p.x/r1,p.z/r2)
    d=vec2((r.norm()-1.0)*ms+rt,ti.abs(p.y)-h)+rr; return min(max(d.x,d.y),0.0)+max(d,0.0).norm()-rr<0
@ti.func
def box(x, y, z, round, cone, unused, p):
    ms=min(x,min(y,z));rr=ms*round;rt=mix(cone*(max(ms-rr,0)),0,float(y-p.y)*0.5/y);q=ti.abs(p)-vec3(x-rt,y,z-rt)+rr
    return ti.max(q, 0.0).norm() + ti.min(ti.max(q.x, ti.max(q.y, q.z)), 0.0) - rr< 0
@ti.func
def tri(r1, h, r2, round_unused, cone, vertex, p):
    r = vec3(p.x/r1, p.y, p.z/r2);rt=mix(1.0-cone,1.0,float(h-p.y)*0.5/h);r.z+=(r.x+1)*mix(-0.577, 0.577, vertex)
    q = ti.abs(r); return max(q.y-h,max(q.z*0.866025+r.x*0.5,-r.x)-0.5*rt)< 0
@ti.func
def make(func: ti.template(), p1, p2, p3, p4, p5, p6, pos, dir, up, color, mat, mode):
    max_r = 2 * int(max(p3,max(p1, p2))); dir = normalize(dir); up = normalize(cross(cross(dir, up), dir))
    for i,j,k in ti.ndrange((-max_r,max_r),(-max_r,max_r),(-max_r,max_r)): 
        xyz = proj_plane(vec3(0.0,0.0,0.0), dir, up, vec3(i,j,k))
        if func(p1,p2,p3,p4,p5,p6,xyz):
            if mode == 0: scene.set_voxel(pos + vec3(i,j,k), mat, color) # additive
            if mode == 1: scene.set_voxel(pos + vec3(i,j,k), 0, color) # subtractive
            if mode == 2 and scene.get_voxel(pos + vec3(i,j,k))[0] > 0: scene.set_voxel(pos + vec3(i,j,k), mat, color)
@ti.kernel
def initialize_voxels():
    make(box,5.0,15.0,8.0,0.1,0.0,0.0,vec3(9,-24,0),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(225,149,35),1,0)
    make(box,2.0,25.0,1.0,0.0,0.0,0.0,vec3(9,-34,-8),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,2.0,25.0,1.0,0.0,0.0,0.0,vec3(9,-34,8),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(cyli,6.0,2.0,3.0,0.1,0.0,0.0,vec3(6,-61,-5),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(225,177,71),1,0)
    make(box,4.0,10.0,3.0,0.0,0.0,0.0,vec3(9,-49,-5),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(225,149,35),1,0)
    make(cyli,6.0,2.0,3.0,0.1,0.0,0.0,vec3(6,-61,5),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(225,177,71),1,0)
    make(box,4.0,10.0,3.0,0.0,0.0,0.0,vec3(9,-49,5),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(225,149,35),1,0)
    make(box,2.0,10.0,1.0,0.0,0.0,0.0,vec3(9,-49,2),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,2.0,10.0,1.0,0.0,0.0,0.0,vec3(9,-49,-2),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,8.0,10.0,10.0,0.1,0.0,0.0,vec3(9,1,0),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(234,179,138),1,0)
    make(box,10.0,2.0,12.0,0.0,0.0,0.0,vec3(9,12,0),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,1.5,12.0,12.0,0.0,0.0,0.0,vec3(18,2,0),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,1.0,3.0,12.0,0.0,0.0,0.0,vec3(1,11,0),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,1.0,1.5,1.5,0.0,0.0,0.0,vec3(1,5,-6),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(255,255,255),1,0)
    make(box,1.0,1.5,1.5,0.0,0.0,0.0,vec3(1,5,-4),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,1.0,1.5,1.5,0.0,0.0,0.0,vec3(1,5,6),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(255,255,255),1,0)
    make(box,1.0,1.5,1.5,0.0,0.0,0.0,vec3(1,5,4),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,1.0,1.5,2.5,0.0,0.0,0.0,vec3(1,-2,0),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(255,255,255),1,0)
    make(box,1.0,1.5,2.5,0.0,0.0,0.0,vec3(1,-5,0),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,10.0,3.5,1.0,0.0,0.0,0.0,vec3(9,10,11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,10.0,3.5,1.0,0.0,0.0,0.0,vec3(9,10,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,7.5,2.0,1.0,0.0,0.0,0.0,vec3(11,5,11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,7.5,2.0,1.0,0.0,0.0,0.0,vec3(11,5,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,6.0,2.0,1.0,0.0,0.0,0.0,vec3(13,2,11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,6.0,2.0,1.0,0.0,0.0,0.0,vec3(13,2,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,4.5,2.0,1.0,0.0,0.0,0.0,vec3(14,-1,11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,4.5,2.0,1.0,0.0,0.0,0.0,vec3(14,-1,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,3.0,2.0,1.0,0.0,0.0,0.0,vec3(16,-4,11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,3.0,2.0,1.0,0.0,0.0,0.0,vec3(16,-4,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(0,0,0),1,0)
    make(box,10.0,4.0,3.0,0.0,0.0,0.0,vec3(4,-13,11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,-0.0),rgb(225,149,35),1,0)
    make(box,10.0,2.0,1.0,0.0,0.0,0.0,vec3(4,-13,14),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,-0.0),rgb(0,0,0),1,0)
    make(cyli,3.0,6.0,3.0,0.1,0.3,0.0,vec3(-10,-13,11),vec3(-1.0,-0.0,0.0),vec3(-0.0,1.0,0.0),rgb(234,179,138),1,0)
    make(box,4.0,10.0,3.0,0.0,0.0,0.0,vec3(9,-19,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,-0.0),rgb(225,149,35),1,0)
    make(box,2.0,10.0,1.0,0.0,0.0,0.0,vec3(9,-19,-14),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,-0.0),rgb(0,0,0),1,0)
    make(box,2.5,2.5,2.5,0.1,0.0,0.0,vec3(-16,-13,11),vec3(0.0,1.0,0.0),vec3(1.0,0.0,0.0),rgb(234,179,138),1,0)
    make(cyli,3.0,6.0,3.0,0.1,0.3,0.0,vec3(1,-26,-11),vec3(-1.0,-0.0,0.0),vec3(-0.0,1.0,0.0),rgb(234,179,138),1,0)
    make(box,2.5,2.5,2.5,0.1,0.0,0.0,vec3(-5,-26,-11),vec3(0.0,1.0,0.0),vec3(1.0,0.0,0.0),rgb(234,179,138),1,0)
    make(cyli,1.5,10.0,1.5,0.1,0.0,0.0,vec3(-5,-24,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(225,177,71),1,0)
    make(elli,1.0,1.0,1.0,0.0,0.0,0.0,vec3(-5,-13,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(192,192,192),1,0)
    make(elli,1.0,1.0,1.0,0.0,0.0,0.0,vec3(-5,-11,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(192,192,192),1,0)
    make(elli,1.0,1.0,1.0,0.0,0.0,0.0,vec3(-5,-9,-11),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(192,192,192),1,0)
    make(elli,1.0,1.0,1.0,0.0,0.0,0.0,vec3(-5,-9,-13),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(192,192,192),1,0)
    make(elli,1.0,1.0,1.0,0.0,0.0,0.0,vec3(-4,-10,-14),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(192,192,192),1,0)
    make(elli,1.0,1.0,1.0,0.0,0.0,0.0,vec3(-2,-10,-14),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(192,192,192),1,0)
    make(elli,1.0,1.0,1.0,0.0,0.0,0.0,vec3(-2,-10,-16),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(192,192,192),1,0)
    make(elli,1.0,1.0,1.0,0.0,0.0,0.0,vec3(-1,-11,-17),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(192,192,192),1,0)
    make(cyli,1.5,10.0,1.5,0.1,0.0,0.0,vec3(-1,-22,-17),vec3(-0.0,-1.0,0.0),vec3(-1.0,0.0,0.0),rgb(225,177,71),1,0)
initialize_voxels()
scene.finish()
