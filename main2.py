  # yapf: disable
from scene import Scene
import taichi as ti
from taichi.math import *
scene = Scene(voxel_edges=0.0, exposure=10)
scene.set_floor(-1, (0, 0, 0))
@ti.func
def y_(x, y0, y1):
  for y, z in ti.ndrange((y0, y1), (-1, 1)):
    scene.set_voxel(vec3(x, y, z), 2, vec3(0.1, 0.1, 0.1))
@ti.kernel
def initialize_voxels():
  y_(-40,-40,-39)
  y_(-39,-39,-38)
  y_(-38,-38,-37)
  y_(-37,-37,-35)
  y_(-36,-36,-34)
  y_(-35,-34,-33)
  y_(-34,-33,-32)
  y_(-33,-32,-31)
  y_(-32,-32,-30)
  y_(-31,-31,-30)
  y_(-30,-32,-29)
  y_(-29,-33,-31);y_(-29,-30,-29)
  y_(-28,-33,-32);y_(-28,-30,-28)
  y_(-27,-34,-32);y_(-27,-30,-27)
  y_(-26,-34,-33);y_(-26,-30,-26)
  y_(-25,-37,-36);y_(-25,-35,-33);y_(-25,-30,-28);y_(-25,-27,-26)
  y_(-24,-37,-34);y_(-24,-31,-29);y_(-24,-26,-25)
  y_(-23,-38,-34);y_(-23,-31,-29);y_(-23,-26,-25);y_(-23,-24,-21)
  y_(-22,-39,-35);y_(-22,-31,-30);y_(-22,-26,-24);y_(-22,-21,-19);y_(-22,-17,-15)
  y_(-21,-40,-35);y_(-21,-27,-25);y_(-21,-20,-14);y_(-21,23,29)
  y_(-20,-40,-36);y_(-20,-27,-26);y_(-20,-19,-13);y_(-20,16,26)
  y_(-19,-40,-37);y_(-19,-28,-27);y_(-19,-15,-13);y_(-19,-2,4);y_(-19,6,10);y_(-19,11,18)
  y_(-18,-40,-38);y_(-18,-29,-27);y_(-18,-15,-14);y_(-18,-9,-2);y_(-18,3,9)
  y_(-17,-40,-39);y_(-17,-16,-14);y_(-17,-11,-8);y_(-17,-6,-4);y_(-17,-1,3);y_(-17,4,6)
  y_(-16,-17,-15);y_(-16,-11,-10);y_(-16,-7,-1)
  y_(-15,-16,-15);y_(-15,-12,-11)
  y_(-14,-16,-11)
  y_(-13,-16,-15);y_(-13,-13,-12);y_(-13,6,10)
  y_(-12,-13,-11);y_(-12,4,7)
  y_(-11,2,5);y_(-11,7,8)
  y_(-10,-13,-11);y_(-10,1,5);y_(-10,6,7)
  y_(-9,0,4)
  y_(-8,-17,-15);y_(-8,-1,4);y_(-8,6,8)
  y_(-7,-18,-13);y_(-7,-12,-11);y_(-7,-10,-9);y_(-7,-1,1);y_(-7,2,3)
  y_(-6,-18,-10);y_(-6,0,1)
  y_(-5,-19,-13)
  y_(-4,-19,-15)
  y_(-3,-19,-17)
  y_(-2,-20,-18)
  y_(-1,-21,-19)
  y_(0,-22,-20);y_(0,-7,-3);y_(0,52,54)
  y_(1,-6,-2);y_(1,53,55)
  y_(2,-8,-2);y_(2,53,56)
  y_(3,-10,-4);y_(3,4,8);y_(3,54,56)
  y_(4,-12,-3);y_(4,4,13);y_(4,53,57)
  y_(5,-13,-2);y_(5,2,12);y_(5,51,53)
  y_(6,-15,-3);y_(6,0,11);y_(6,52,54);y_(6,55,56)
  y_(7,-16,13);y_(7,51,52);y_(7,53,54);y_(7,55,56);y_(7,57,58)
  y_(8,-17,15);y_(8,51,52);y_(8,53,54);y_(8,55,56);y_(8,57,58)
  y_(9,-18,15);y_(9,53,58)
  y_(10,-19,14);y_(10,51,58)
  y_(11,-19,15);y_(11,54,58)
  y_(12,-20,14);y_(12,50,52);y_(12,53,58)
  y_(13,-21,7);y_(13,8,11);y_(13,13,15);y_(13,49,51);y_(13,54,58)
  y_(14,-22,7);y_(14,9,11);y_(14,14,19);y_(14,55,58)
  y_(15,-23,6);y_(15,9,11);y_(15,14,16);y_(15,17,20);y_(15,55,58)
  y_(16,-37,-35);y_(16,-33,-30);y_(16,-28,-27);y_(16,-23,6);y_(16,10,11);y_(16,14,15);y_(16,51,54);y_(16,56,58)
  y_(17,-36,-31);y_(17,-28,-27);y_(17,-24,6);y_(17,10,11);y_(17,14,15);y_(17,50,54);y_(17,56,57)
  y_(18,-36,-33);y_(18,-29,-28);y_(18,-25,6);y_(18,10,11);y_(18,14,18);y_(18,52,54);y_(18,55,57)
  y_(19,-35,-29);y_(19,-26,-15);y_(19,-14,6);y_(19,10,11);y_(19,14,17);y_(19,50,56)
  y_(20,-34,-31);y_(20,-26,-14);y_(20,-13,8);y_(20,10,11);y_(20,14,18);y_(20,51,55)
  y_(21,-38,-37);y_(21,-26,-14);y_(21,-12,8);y_(21,10,11);y_(21,14,17);y_(21,50,54)
  y_(22,-38,-37);y_(22,-25,-14);y_(22,-13,7);y_(22,9,11);y_(22,14,16);y_(22,49,53)
  y_(23,-39,-38);y_(23,-26,-14);y_(23,-13,-8);y_(23,-4,6);y_(23,9,10);y_(23,14,16);y_(23,49,52)
  y_(24,-39,-38);y_(24,-26,-14);y_(24,-13,-8);y_(24,-5,6);y_(24,8,9);y_(24,13,17);y_(24,50,52)
  y_(25,-40,-39);y_(25,-26,-14);y_(25,-13,-8);y_(25,-6,8);y_(25,13,17);y_(25,49,50)
  y_(26,-40,-39);y_(26,-25,-18);y_(26,-17,-14);y_(26,-12,-8);y_(26,-6,7);y_(26,12,17);y_(26,50,51)
  y_(27,-25,-19);y_(27,-17,-14);y_(27,-12,-8);y_(27,-6,8);y_(27,12,17)
  y_(28,-24,-19);y_(28,-17,-15);y_(28,-13,-9);y_(28,-6,9);y_(28,10,16)
  y_(29,-23,-14);y_(29,-12,-9);y_(29,-7,15)
  y_(30,-19,-14);y_(30,-12,-9);y_(30,-7,16)
  y_(31,-15,-14);y_(31,-13,-8);y_(31,-7,6);y_(31,8,15)
  y_(32,-14,-13);y_(32,-12,5);y_(32,9,14)
  y_(33,-8,5);y_(33,10,14)
  y_(34,-6,6);y_(34,11,15);y_(34,47,48)
  y_(35,-5,4);y_(35,5,7);y_(35,8,9);y_(35,12,16);y_(35,47,48)
  y_(36,-3,5);y_(36,6,7);y_(36,9,10);y_(36,13,14)
  y_(37,4,6);y_(37,9,10);y_(37,14,15)
  y_(38,9,11);y_(38,14,17);y_(38,44,45)
  y_(39,10,11)
  y_(40,42,43)
  y_(44,35,37)

initialize_voxels()
scene.finish()