import os
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pymxs
from pymxs import runtime as rt

# Animation flags
ANIM_FLAG_NONE = 0x00
ANIM_FLAG_LOOPED = 0x01
ANIM_FLAG_TRANSLATION = 0x02
ANIM_FLAG_UNK1 = 0x04
ANIM_FLAG_UNK2 = 0x08

# Fixed record sizes
BONE_SIZE = 100
FRAME_TABLE_ENTRY_SIZE = 12
OFFSET_ENTRY_V1_SIZE = 24

# Global matrices
max_mat = rt.matrix3(
    rt.point3(0, 0, 1),
    rt.point3(0, 1, 0),
    rt.point3(-1, 0, 0),
    rt.point3(0, 0, 0)
)

correction = rt.Matrix3(
    rt.point3(1, 0, 0),
    rt.point3(0, 0, 1),
    rt.point3(0, -1, 0),
    rt.point3(0, 0, 0)
)


class BinaryWriter:
    def __init__(self):
        self.data = bytearray()

    def write_int32(self, v: int):
        self.data += struct.pack("<i", v)

    def write_uint32(self, v: int):
        self.data += struct.pack("<I", v)

    def write_int16(self, v: int):
        self.data += struct.pack("<h", v)

    def write_uint16(self, v: int):
        self.data += struct.pack("<H", v)

    def write_float(self, v: float):
        self.data += struct.pack("<f", v)

    def write_uint8(self, v: int):
        self.data += struct.pack("<B", v)

    def write_bytes(self, b: bytes):
        self.data += b

    def write_fixed_string(self, s: str, length: int):
        enc = s.encode("ascii", "ignore")[:length]
        self.data += enc.ljust(length, b"\0")

    def getvalue(self) -> bytes:
        return bytes(self.data)


def matrix3_to_floats(m) -> Tuple[float, ...]:
    return (
        m.row1.x, m.row1.y, m.row1.z,
        m.row2.x, m.row2.y, m.row2.z,
        m.row3.x, m.row3.y, m.row3.z
    )


@dataclass
class AnimationHeader:
    Version: int = 1
    HeaderLen: int = 80
    FPS: int = 30
    NumFrames: int = 2
    AnimationFlags: int = ANIM_FLAG_LOOPED | ANIM_FLAG_TRANSLATION
    NumBones: int = 0
    BoneOffset: int = 0
    FrameDataOffset: int = 0
    Unknown1: int = 0
    NumEventBlocks: int = 0
    Unknown3: int = 8
    sizeofBoneData: int = BONE_SIZE
    sizeofFrameRecord: int = FRAME_TABLE_ENTRY_SIZE
    sizeofRotation: int = 16
    PosOffsetsFrameLen: int = 1
    NumOffsets: int = 0
    rootOffsetsOffset: int = 0
    Unknown6: int = 1
    Unknown7: int = 0
    Unknown8: int = 0

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


@dataclass
class Bone:
    name: str = "UNNAMED"
    unknown1: int = 0
    unknown2: int = 0
    unknown3: int = 0
    BoneNum: int = 0
    NumChildren: int = 0
    ChildOffset: int = 0
    ParentOffset: int = 0
    BoneLength: float = 0.0
    Position: any = None  # rt.point3
    Transform: any = None  # rt.Matrix3
    parent: Optional["Bone"] = None
    children: List["Bone"] = field(default_factory=list)

    def __post_init__(self):
        if self.Position is None:
            self.Position = rt.point3(0.0, 0.0, 0.0)
        if self.Transform is None:
            self.Transform = rt.Matrix3(
                rt.point3(1, 0, 0),
                rt.point3(0, 1, 0),
                rt.point3(0, 0, 1),
                rt.point3(0, 0, 0)
            )

    def write(self, w: BinaryWriter):
        enc_name = self.name.encode("ascii", "ignore")[:31] + b"\0"
        w.write_bytes(enc_name.ljust(32, b"\0"))
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
        for val in matrix3_to_floats(self.Transform):
            w.write_float(val)


@dataclass
class FrameTableEntry:
    NumKeys: int = 0
    frame_length_ptr: int = 0
    quat_ptr: int = 0

    def write(self, w: BinaryWriter):
        w.write_int32(self.NumKeys)
        w.write_int32(self.frame_length_ptr)
        w.write_int32(self.quat_ptr)


@dataclass
class OffsetEntryV1:
    vel: any = None  # rt.point3
    bottom: float = 0.0
    top: float = 0.0
    event: int = 0

    def __post_init__(self):
        if self.vel is None:
            self.vel = rt.point3(0.0, 0.0, 0.0)

    def write(self, w: BinaryWriter):
        w.write_float(self.vel.x)
        w.write_float(self.vel.y)
        w.write_float(self.vel.z)
        w.write_float(self.bottom)
        w.write_float(self.top)
        w.write_int32(self.event)


@dataclass
class Channel:
    frame_lengths: List[int] = field(default_factory=list)
    rotations: List[Tuple[float, float, float, float]] = field(default_factory=list)
    translations: Optional[List[Tuple[float, float, float]]] = None


@dataclass
class BadFile:
    header: AnimationHeader = field(default_factory=AnimationHeader)
    bones: List[Bone] = field(default_factory=list)
    channels: List[Channel] = field(default_factory=list)
    offsets: List[OffsetEntryV1] = field(default_factory=list)

    def __post_init__(self):
        while len(self.channels) < len(self.bones):
            self.channels.append(Channel())

    def write(self, filename: str):
        self.header.NumBones = len(self.bones)
        for i, b in enumerate(self.bones):
            b.BoneNum = i
            b.NumChildren = len(b.children)

        w = BinaryWriter()
        # Reserve header space
        w.write_bytes(b"\x00" * self.header.HeaderLen)
        # Reserve frame table entries space
        frame_table_offset = len(w.data)
        w.write_bytes(b"\x00" * (FRAME_TABLE_ENTRY_SIZE * self.header.NumBones))

        bone_framelength_ptrs = []
        bone_quat_ptrs = []
        for ch in self.channels:
            fl_ptr = len(w.data)
            for fl in ch.frame_lengths:
                w.write_uint16(fl)
            quat_ptr = len(w.data)
            for (x, y, z, qq) in ch.rotations:
                w.write_float(x)
                w.write_float(y)
                w.write_float(z)
                w.write_float(qq)
            bone_framelength_ptrs.append(fl_ptr)
            bone_quat_ptrs.append(quat_ptr)

        rootOffsetsOffset = 0
        if self.header.NumOffsets > 0 and self.offsets:
            rootOffsetsOffset = len(w.data)
            for off in self.offsets:
                off.write(w)

        bone_offset = len(w.data)
        for b in self.bones:
            if b.parent:
                b.ParentOffset = bone_offset + self.bones.index(b.parent) * BONE_SIZE
            else:
                b.ParentOffset = -1
            if b.children:
                b.ChildOffset = bone_offset + self.bones.index(b.children[0]) * BONE_SIZE
            else:
                b.ChildOffset = -1

        for b in self.bones:
            b.write(w)

        if self.header.AnimationFlags & ANIM_FLAG_TRANSLATION:
            for f in range(self.header.NumFrames + 1):
                for i in range(self.header.NumBones):
                    tx, ty, tz = self.channels[i].translations[f]
                    w.write_float(tx)
                    w.write_float(ty)
                    w.write_float(tz)

        for i, ch in enumerate(self.channels):
            num_keys = len(ch.frame_lengths)
            fte = FrameTableEntry(num_keys, bone_framelength_ptrs[i], bone_quat_ptrs[i])
            start = frame_table_offset + i * FRAME_TABLE_ENTRY_SIZE
            subw = BinaryWriter()
            fte.write(subw)
            w.data[start:start + FRAME_TABLE_ENTRY_SIZE] = subw.data

        self.header.BoneOffset = bone_offset
        self.header.FrameDataOffset = frame_table_offset
        self.header.rootOffsetsOffset = rootOffsetsOffset

        hw = BinaryWriter()
        self.header.write(hw)
        w.data[0:self.header.HeaderLen] = hw.data

        with open(filename, "wb") as f:
            f.write(w.getvalue())


def collect_all_bones(node):
    bones = []
    if rt.isValidNode(node) and ("BN" in node.name or "bn" in node.name):
        bones.append(node)
    for child in node.children:
        bones.extend(collect_all_bones(child))
    return bones


def extract_bone_data(bone_node) -> Bone:
    rot_mat = rt.MyUtilsInstance.ConvertMxsType(bone_node.objecttransform.rotation, rt.Matrix3)
    cs = rt.matrix3(
        rt.point3(1, 0, 0),
        rt.point3(0, 1, 0),
        rt.point3(0, 0, 1),
        bone_node.transform.position
    )
    result = rot_mat * cs

    if bone_node.parent is not None:
        local_tm = bone_node.transform * rt.inverse(bone_node.parent.transform)
    else:
        local_tm = bone_node.transform

    start_point = bone_node.transform.position
    end_point = start_point + bone_node.transform.row3 * bone_node.length
    bone_length = rt.distance(start_point, end_point)
    pos = rt.point3(local_tm.position.x, local_tm.position.y, local_tm.position.z)
    bone_name = bone_node.name[:31]

    if "BN01" in bone_name:
        pos = rt.point3(0, 0, 0)

    return Bone(
        name=bone_name,
        BoneLength=bone_length,
        Position=pos * max_mat,
        Transform=rt.inverse(max_mat) * result * rt.inverse(correction)
    )


def extract_animation_frames(bad_file, bone_nodes, start_frame, end_frame, fps, translated=True, initial_bones=None):
    num_frames = end_frame - start_frame + 1
    bone_channels = []
    temp_offsets = []

    for i, bone_node in enumerate(bone_nodes):
        frame_lengths = [1] * num_frames
        rotations = []
        translations = []
        initial_pos = initial_bones[i].Position * rt.inverse(max_mat)

        for f in range(num_frames):
            cf = start_frame + f
            with pymxs.attime(cf):
                rot_mat = rt.MyUtilsInstance.ConvertMxsType(bone_node.objecttransform.rotation, rt.Matrix3)
                cs = rt.matrix3(
                    rt.point3(1, 0, 0),
                    rt.point3(0, 1, 0),
                    rt.point3(0, 0, 1),
                    bone_node.transform.position
                )
                result = rot_mat * cs
                rot_quat = rt.MyUtilsInstance.ConvertMxsType(
                    rt.inverse(max_mat) * result * rt.inverse(correction), rt.quat
                )
                rot_quat = rt.inverse(rot_quat)
                rotations.append((rot_quat.x, rot_quat.y, rot_quat.z, rot_quat.w))

                if translated:
                    current_tm = bone_node.transform.position
                    parent = bone_node.parent if bone_node.parent else bone_node
                    pos_in_parent = current_tm * rt.inverse(parent.transform)
                    ip = initial_pos * parent.transform
                    dist = rt.distance(initial_pos, pos_in_parent)
                    if dist > 0.000001:
                        diff_pos = (rt.point3(
                            current_tm.x - ip.x,
                            current_tm.y - ip.y,
                            current_tm.z - ip.z
                        )) * rt.inverse(correction)
                        translations.append((diff_pos.x, diff_pos.y, diff_pos.z))
                    else:
                        translations.append((0, 0, 0))

            if i == 0:
                if cf != end_frame:
                    with pymxs.attime(cf):
                        curr = bone_node.transform.position
                    with pymxs.attime(cf + 1):
                        nxt = bone_node.transform.position
                    vel = (nxt - curr) * rt.inverse(correction)
                else:
                    vel = temp_offsets[0].vel
                temp_offsets.append(OffsetEntryV1(vel, bottom=0, top=0, event=0))

        ch = Channel(frame_lengths, rotations, translations if translated else None)
        bone_channels.append(ch)

    bad_file.offsets = temp_offsets
    return bone_channels


def find_root_bones(nodes):
    root_bones = []
    for node in nodes:
        if rt.isValidNode(node) and ("BN01" in node.name or "bn01" in node.name):
            if node.parent is None or not rt.isValidNode(node.parent) or not ("BN01" in node.parent.name or "bn01" in node.parent.name):
                root_bones.append(node)
    return root_bones


def export_bad(name, start_frame, end_frame, flags, output_path):
    print("name", name)
    print("start_frame", start_frame)
    print("end_frame", end_frame)
    print("flags", flags)
    print("output_path", output_path)
    filename = output_path
    all_nodes = rt.objects
    root_bones = find_root_bones(all_nodes)

    if not root_bones:
        print("Error: No root bones found in the scene.")
        return
    if len(root_bones) > 1:
        print("Error: Multiple root bones found in the scene:")
        for bone in root_bones:
            print(f"- {bone.name}")
        print("Please ensure there is only one root bone.")
        return

    root_bone = root_bones[0]
    print(f"Using root bone: {root_bone.name}")

    all_bones = collect_all_bones(root_bone)
    all_bones.sort(key=lambda b: b.name)

    fps = rt.frameRate

    bad_file = BadFile()
    bad_file.header.NumBones = len(all_bones)
    bad_file.header.NumFrames = end_frame - start_frame
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
            parent_index = all_bones.index(bn.parent)
            b.parent = bad_file.bones[parent_index]
            b.parent.children.append(b)

    channels = extract_animation_frames(
        bad_file, all_bones, start_frame, end_frame, fps,
        translated=bool(flags & ANIM_FLAG_TRANSLATION),
        initial_bones=bones_list
    )
    bad_file.channels = channels
    bad_file.header.NumOffsets = len(bad_file.offsets)

    bad_file.write(filename)
    print(f"Exported .bad file to {filename}")


def show_opennova_ui():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ui_path = os.path.join(script_dir, "ui.ms")
    myutils_path = os.path.join(script_dir, "utils.ms")
    rt.FileIn(myutils_path)
    rt.FileIn(ui_path)
    rt.createDialog(rt.OpenNovaRoll)


# Launch the UI
show_opennova_ui()
