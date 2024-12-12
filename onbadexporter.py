import struct
import sys
from contextlib import contextmanager
import pymxs
from pymxs import runtime as rt

ANIM_FLAG_NONE        = 0x00
ANIM_FLAG_LOOPED      = 0x01
ANIM_FLAG_TRANSLATION = 0x02
ANIM_FLAG_UNK1        = 0x04
ANIM_FLAG_UNK2        = 0x08

class BinaryWriter:
    def __init__(self):
        self.data = bytearray()

    def write_int32(self, v):
        self.data += struct.pack("<i", v)

    def write_uint32(self, v):
        self.data += struct.pack("<I", v)

    def write_int16(self, v):
        self.data += struct.pack("<h", v)

    def write_uint16(self, v):
        self.data += struct.pack("<H", v)

    def write_float(self, v):
        self.data += struct.pack("<f", v)

    def write_uint8(self, v):
        self.data += struct.pack("<B", v)

    def write_bytes(self, b):
        self.data += b

    def write_fixed_string(self, s, length):
        enc = s.encode('ascii', 'ignore')[:length]
        enc = enc + b'\0' * (length - len(enc))
        self.data += enc

    def getvalue(self):
        return bytes(self.data)


class BinaryReader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read_int32(self):
        v = struct.unpack_from("<i", self.data, self.pos)[0]
        self.pos += 4
        return v

    def read_uint32(self):
        v = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return v

    def read_int16(self):
        v = struct.unpack_from("<h", self.data, self.pos)[0]
        self.pos += 2
        return v

    def read_uint16(self):
        v = struct.unpack_from("<H", self.data, self.pos)[0]
        self.pos += 2
        return v

    def read_float(self):
        v = struct.unpack_from("<f", self.data, self.pos)[0]
        self.pos += 4
        return v

    def read_uint8(self):
        v = self.data[self.pos]
        self.pos += 1
        return v

    def read_bytes(self, length):
        b = self.data[self.pos:self.pos+length]
        self.pos += length
        return b

def matrix3_to_floats(m):
    return (m.row1.x, m.row1.y, m.row1.z,
            m.row2.x, m.row2.y, m.row2.z,
            m.row3.x, m.row3.y, m.row3.z)

def floats_to_matrix3(floats):
    return rt.Matrix3(
        rt.point3(floats[0], floats[1], floats[2]),
        rt.point3(floats[3], floats[4], floats[5]),
        rt.point3(floats[6], floats[7], floats[8]),
        rt.point3(0,0,0)
    )

class AnimationHeader:
    size = 80
    def __init__(self,
                 Version=1,
                 HeaderLen=80,
                 FPS=30,
                 NumFrames=2,
                 AnimationFlags=ANIM_FLAG_LOOPED | ANIM_FLAG_TRANSLATION,
                 NumBones=0,
                 BoneOffset=0,
                 FrameDataOffset=0,
                 Unknown1=0,
                 NumEventBlocks=0,
                 Unknown3=8,
                 sizeofBoneData=100,
                 sizeofFrameRecord=12,
                 sizeofRotation=16,
                 PosOffsetsFrameLen=1,
                 NumOffsets=0,
                 rootOffsetsOffset=0,
                 Unknown6=1,
                 Unknown7=0,
                 Unknown8=0):
        self.Version = Version
        self.HeaderLen = HeaderLen
        self.FPS = FPS
        self.NumFrames = NumFrames
        self.AnimationFlags = AnimationFlags
        self.NumBones = NumBones
        self.BoneOffset = BoneOffset
        self.FrameDataOffset = FrameDataOffset
        self.Unknown1 = Unknown1
        self.NumEventBlocks = NumEventBlocks
        self.Unknown3 = Unknown3
        self.sizeofBoneData = sizeofBoneData
        self.sizeofFrameRecord = sizeofFrameRecord
        self.sizeofRotation = sizeofRotation
        self.PosOffsetsFrameLen = PosOffsetsFrameLen
        self.NumOffsets = NumOffsets
        self.rootOffsetsOffset = rootOffsetsOffset
        self.Unknown6 = Unknown6
        self.Unknown7 = Unknown7
        self.Unknown8 = Unknown8

    def write(self, w: BinaryWriter):
        w.write_int32(self.Version)
        w.write_int32(self.HeaderLen)
        w.write_int32(self.FPS)
        w.write_int32(self.NumFrames)
        w.write_int32(self.AnimationFlags)
        w.write_int32(self.NumBones)
        w.write_int32(self.BoneOffset)
        w.write_int32(self.FrameDataOffset)
        w.write_int32(self.Unknown1)
        w.write_int32(self.NumEventBlocks)
        w.write_int32(self.Unknown3)
        w.write_int32(self.sizeofBoneData)
        w.write_int32(self.sizeofFrameRecord)
        w.write_int32(self.sizeofRotation)
        w.write_int32(self.PosOffsetsFrameLen)
        w.write_int32(self.NumOffsets)
        w.write_int32(self.rootOffsetsOffset)
        w.write_int32(self.Unknown6)
        w.write_int32(self.Unknown7)
        w.write_int32(self.Unknown8)

class Bone:
    size = 100
    def __init__(self, name="UNNAMED",
                 unknown1=0, unknown2=0, unknown3=0, BoneNum=0,
                 NumChildren=0, ChildOffset=0, ParentOffset=0, BoneLength=0.0,
                 Position=None,
                 Transform=None):
        if Position is None:
            Position = rt.point3(0.0,0.0,0.0)
        if Transform is None:
            Transform = rt.Matrix3(rt.point3(1,0,0),
                                   rt.point3(0,1,0),
                                   rt.point3(0,0,1),
                                   rt.point3(0,0,0))
        self.name = name
        self.unknown1 = unknown1
        self.unknown2 = unknown2
        self.unknown3 = unknown3
        self.BoneNum = BoneNum
        self.NumChildren = NumChildren
        self.ChildOffset = ChildOffset
        self.ParentOffset = ParentOffset
        self.BoneLength = BoneLength
        self.Position = Position
        self.Transform = Transform
        self.parent = None
        self.children = []

    def write(self, w: BinaryWriter):
        enc_name = self.name.encode('ascii','ignore')[:31] + b'\0'
        enc_name = enc_name.ljust(32, b'\0')
        w.write_bytes(enc_name)
        w.write_uint8(self.unknown1)
        w.write_uint8(self.unknown2)
        w.write_uint8(self.unknown3)
        w.write_uint8(self.BoneNum)
        w.write_int32(self.NumChildren)
        w.write_int32(self.ChildOffset)
        w.write_int32(self.ParentOffset)
        w.write_float(self.BoneLength)
        w.write_float(self.Position.x)
        w.write_float(self.Position.y)
        w.write_float(self.Position.z)

        floats = matrix3_to_floats(self.Transform)
        for val in floats:
            w.write_float(val)

class FrameTableEntry:
    size = 12
    def __init__(self, NumFrames=0, Padding=0, frame_length_ptr=0, quat_ptr=0):
        self.NumFrames = NumFrames
        self.Padding = Padding
        self.frame_length_ptr = frame_length_ptr
        self.quat_ptr = quat_ptr

    def write(self, w: BinaryWriter):
        w.write_uint16(self.NumFrames)
        w.write_uint16(self.Padding)
        w.write_int32(self.frame_length_ptr)
        w.write_int32(self.quat_ptr)

class OffsetEntryV1:
    size = 24
    def __init__(self, vel=None, bottom=0.0, top=0.0, event=0):
        if vel is None:
            vel = rt.point3(0.0,0.0,0.0)
        self.vel = vel
        self.bottom = bottom
        self.top = top
        self.event = event

    def write(self, w: BinaryWriter):
        w.write_float(self.vel.x)
        w.write_float(self.vel.y)
        w.write_float(self.vel.z)
        w.write_float(self.bottom)
        w.write_float(self.top)
        w.write_int32(self.event)

class Channel:
    def __init__(self, frame_lengths, rotations, translations=None):
        self.frame_lengths = frame_lengths
        self.rotations = rotations
        self.translations = translations

class BadFile:
    def __init__(self, header=None, bones=None, channels=None, offsets=None):
        self.header = header or AnimationHeader()
        self.bones = bones or []
        self.channels = channels or []
        self.offsets = offsets or []
        while len(self.channels) < len(self.bones):
            self.channels.append(Channel([],[],None))

    def write(self, filename):
        self.header.NumBones = len(self.bones)
        for i,b in enumerate(self.bones):
            b.BoneNum = i
            b.NumChildren = len(b.children)

        w = BinaryWriter()
        w.write_bytes(b'\x00'*80)  # header placeholder
        frame_table_offset = len(w.data)
        w.write_bytes(b'\x00'*(FrameTableEntry.size*self.header.NumBones))

        bone_framelength_ptrs = []
        bone_quat_ptrs = []
        for ch in self.channels:
            fl_ptr = len(w.data)
            for fl in ch.frame_lengths:
                w.write_uint16(fl)
            quat_ptr = len(w.data)
            for (x,y,z,qq) in ch.rotations:
                w.write_float(x)
                w.write_float(y)
                w.write_float(z)
                w.write_float(-qq)
            bone_framelength_ptrs.append(fl_ptr)
            bone_quat_ptrs.append(quat_ptr)

        rootOffsetsOffset = 0
        if self.header.NumOffsets > 0 and self.offsets:
            rootOffsetsOffset = len(w.data)
            for off in self.offsets:
                off.write(w)

        bone_offset = len(w.data)
        for b in self.bones:
            b.write(w)

        if self.header.AnimationFlags & ANIM_FLAG_TRANSLATION:
            for f in range(self.header.NumFrames):               # frame-major
                for i in range(self.header.NumBones):
                    tx, ty, tz = self.channels[i].translations[f]
                    w.write_float(tx)
                    w.write_float(ty)
                    w.write_float(tz)

        for i,b in enumerate(self.bones):
            if b.parent:
                b.ParentOffset = bone_offset + self.bones.index(b.parent)*Bone.size
            else:
                b.ParentOffset = 0
            if b.children:
                b.ChildOffset = bone_offset + self.bones.index(b.children[0])*Bone.size
            else:
                b.ChildOffset = 0
            start = bone_offset + i*Bone.size
            subw = BinaryWriter()
            b.write(subw)
            w.data[start:start+Bone.size] = subw.data

        for i,ch in enumerate(self.channels):
            fte = FrameTableEntry(self.header.NumFrames,0,bone_framelength_ptrs[i],bone_quat_ptrs[i])
            start = frame_table_offset + i*FrameTableEntry.size
            subw = BinaryWriter()
            fte.write(subw)
            w.data[start:start+FrameTableEntry.size] = subw.data

        self.header.BoneOffset = bone_offset
        self.header.FrameDataOffset = frame_table_offset
        self.header.rootOffsetsOffset = rootOffsetsOffset

        hw = BinaryWriter()
        self.header.write(hw)
        w.data[0:80] = hw.data

        with open(filename,"wb") as f:
            f.write(w.getvalue())

@contextmanager
def set_ref_coordsys(cs_new):
    original_cs = rt.GetRefCoordSys()
    try:
        rt.setRefCoordSys(cs_new)
        yield
    finally:
        rt.setRefCoordSys(original_cs)

def flatten_array(arr):
    result = []
    for item in arr:
        if isinstance(item, list):
            result += flatten_array(item)
        else:
            result.append(item)
    return result

def collect_all_bones(node):
    bones = []
    if rt.isValidNode(node) and ("BN" in node.name):
        bones.append(node)
        for child in node.children:
            bones.append(collect_all_bones(child))
    return flatten_array(bones)

def extract_bone_data(bone_node):
    # Keep as original: initial local space extraction
    correction = rt.Matrix3(rt.point3(1,0,0),
                            rt.point3(0,0,1),
                            rt.point3(0,-1,0),
                            rt.point3(0,0,0))
    with set_ref_coordsys(rt.Name('local')):
        rot_mat = rt.MyUtilsInstance.ConvertMxsType(bone_node.objecttransform.rotation, rt.Matrix3)*rt.inverse(correction)

    # initial local position
    if bone_node.parent is not None:
        local_tm = bone_node.transform * rt.inverse(bone_node.parent.transform)
    else:
        local_tm = bone_node.transform

    pos = rt.point3(local_tm.position.x, local_tm.position.y, local_tm.position.z)
    bone = Bone(name=bone_node.name[:31],
                BoneLength=0.0,
                Position=pos,
                Transform=rot_mat)
    bone.NumChildren = len(bone_node.children)
    return bone

def extract_animation_frames(bad_file, bone_nodes, start_frame, end_frame, fps, translated=True, initial_bones=None):
    num_frames = (end_frame - start_frame + 1)
    correction = rt.Matrix3(rt.point3(1,0,0),
                            rt.point3(0,0,1),
                            rt.point3(0,-1,0),
                            rt.point3(0,0,0))
    bone_channels = []
    temp_offsets = []

    for i, bone_node in enumerate(bone_nodes):
        frame_lengths = [1]*num_frames
        rotations = []
        translations = []
        initial_pos = initial_bones[i].Position

        for f in range(num_frames):
            cf = start_frame + f
            # Rotation
            with pymxs.attime(cf):
                with set_ref_coordsys(rt.Name('local')):
                    rot_mat = rt.MyUtilsInstance.ConvertMxsType(
                        bone_node.objecttransform.rotation,
                        rt.Matrix3
                    ) * rt.inverse(correction)
                rot_quat = rt.MyUtilsInstance.ConvertMxsType(rot_mat, rt.quat)
                rot_quat = rt.normalize(rot_quat)
                rotations.append((rot_quat.x, rot_quat.y, rot_quat.z, rot_quat.w))

            # Translation
            with pymxs.attime(cf):
                if translated:
                    current_tm = bone_node.transform.position
                    parent = bone_node.parent if bone_node.parent else bone_nodes[0]
                    ip = initial_pos * parent.transform
                    diff_pos = (rt.point3(current_tm.x - ip.x,
                                          current_tm.y - ip.y,
                                          current_tm.z - ip.z) 
                                * rt.inverse(correction))
                    translations.append((diff_pos.x, diff_pos.y, diff_pos.z))

            # Some sort of root motion animation?
            if i == 0:
                with pymxs.attime(cf):
                    curr = bone_node.transform.position
                with pymxs.attime(cf+1):
                    nxt = bone_node.transform.position
                vel = ((nxt - curr))*rt.inverse(correction)

                # No idea
                min_z_frame = 0
                max_z_frame = 0

                # Store offsets for this frame (per-frame top/bottom)
                temp_offsets.append(OffsetEntryV1(vel, bottom=min_z_frame, top=max_z_frame, event=0))

        ch = Channel(frame_lengths, rotations, translations if translated else None)
        bone_channels.append(ch)

    bad_file.offsets = temp_offsets
    return bone_channels

def export_bad():
    filename = rt.OpenNovaRoll.et_outputPath.text
    root_bone = rt.getNodeByName("BN01")
    if root_bone is None:
        print("No root bone 'BN01' found.")
        return

    all_bones = collect_all_bones(root_bone)
    all_bones.sort(key=lambda b: b.name)

    start_frame = rt.OpenNovaRoll.spn_startFrame.value
    end_frame = rt.OpenNovaRoll.spn_endFrame.value
    num_frames = end_frame - start_frame# + 1
    fps = rt.frameRate
    dump_to_console = rt.OpenNovaRoll.chk_console.checked

    flags = 0
    if rt.OpenNovaRoll.chk_looped.checked:
        flags |= ANIM_FLAG_LOOPED
    if rt.OpenNovaRoll.chk_trans.checked:
        flags |= ANIM_FLAG_TRANSLATION
    if rt.OpenNovaRoll.chk_bit3.checked:
        flags |= ANIM_FLAG_UNK1
    if rt.OpenNovaRoll.chk_bit4.checked:
        flags |= ANIM_FLAG_UNK2

    bad_file = BadFile()
    bad_file.header.NumBones = len(all_bones)
    bad_file.header.NumFrames = num_frames
    bad_file.header.FPS = fps
    bad_file.header.AnimationFlags = flags

    with pymxs.attime(start_frame):
        bones_list = []
        for i, bn in enumerate(all_bones):
            bone = extract_bone_data(bn)
            bone.BoneNum = i
            bones_list.append(bone)
    bad_file.bones = bones_list

    for i, bn in enumerate(all_bones):
        b = bad_file.bones[i]
        if bn.parent is not None and bn.parent in all_bones:
            parentIndex = all_bones.index(bn.parent)
            b.parent = bad_file.bones[parentIndex]
            b.parent.children.append(b)

    channels = extract_animation_frames(bad_file, all_bones, start_frame, end_frame, fps, translated=(flags & ANIM_FLAG_TRANSLATION), initial_bones=bones_list)
    bad_file.channels = channels

    bad_file.header.NumOffsets = num_frames

    bad_file.write(filename)
    print(f"Exported .bad file to {filename}")

    if dump_to_console:
        pass

def show_opennova_ui():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ui_path = os.path.join(script_dir, "ui.ms")
    myutils_path = os.path.join(script_dir, "utils.ms")
    rt.FileIn(myutils_path)
    rt.FileIn(ui_path)
    rt.createDialog(rt.OpenNovaRoll)

show_opennova_ui()
