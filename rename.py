import pymxs

def rename_bones(root_bone_name, prefix='BN', start_index=1):
    """
    Renames all bones under the specified root bone in the scene.

    Parameters:
    - root_bone_name (str): The name of the root bone.
    - prefix (str): The prefix for the new bone names. Default is 'BN'.
    - start_index (int): The starting index for numbering. Default is 0.
    """
    rt = pymxs.runtime

    # Find the root bone by name
    root_bone = rt.getNodeByName(root_bone_name)
    if not root_bone:
        print(f"Error: Root bone '{root_bone_name}' not found in the scene.")
        return

    # Verify that the found node is a bone
    #3if not rt.isKindOf(root_bone, rt.BoneGeometry) and not rt.isKindOf(root_bone, rt.Biped_Object):
    #    print(f"Error: Node '{root_bone_name}' is not a bone.")
    #    return

    # Initialize list to store bones to rename
    bones_to_rename = []

    def collect_bones(bone):
        """
        Recursively collects all descendant bones of the given bone.
        """
        bones_to_rename.append(bone)
        for child in bone.children:
            if rt.isKindOf(child, rt.BoneGeometry) or rt.isKindOf(root_bone, rt.Biped_Object):
                collect_bones(child)

    # Start collecting bones from the root bone
    collect_bones(root_bone)

    # Rename each collected bone
    for idx, bone in enumerate(bones_to_rename, start=start_index):
        new_name = f"{prefix}{idx:02}"
        print(f"Renaming bone '{bone.name}' to '{new_name}'")
        bone.name = new_name

    print("Bone renaming completed successfully.")

# =======================e
# === Example Usage ====
# =======================

# Replace 'RootBoneName' with the actual name of your root bone
root_bone_name = 'mixamorig:Hips'  # <-- Change this to your root bone's name

# Call the function to rename bones
rename_bones(root_bone_name)
