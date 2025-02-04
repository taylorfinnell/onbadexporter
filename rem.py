import pymxs
rt = pymxs.runtime

def reset_materials(color=(128, 128, 128)):
    """
    Removes all materials from scene geometry and applies a simple diffuse material
    with the specified color.
    
    Args:
        color (tuple): RGB color values from 0-255. Defaults to medium gray.
    """
    # Create a new standard material with the specified diffuse color
    simple_mat = rt.StandardMaterial()
    simple_mat.diffuse = rt.Color(color[0], color[1], color[2])
    simple_mat.name = "SimpleDiffuse"
    
    # Get all geometry in the scene
    all_objects = rt.objects
    
    # Counter for modified objects
    modified_count = 0
    
    # Iterate through all objects
    for obj in all_objects:
        try:
            # Check if object can have materials
            if hasattr(obj, 'material'):
                obj.material = simple_mat
                modified_count += 1
        except Exception as e:
            print(f"Failed to modify {obj.name}: {str(e)}")
    
    print(f"Modified {modified_count} objects with new simple material")

def main():
    # Create undo point before making changes
    #rt.undo(True)
    
    try:
        # You can customize the color here (RGB values 0-255)
        reset_materials(color=(200, 200, 200))  # Light gray
        
        # Refresh viewport to see changes
        rt.redrawViews()
        print("Material reset completed successfully")
        
    except Exception as e:
        rt.undo(False)  # Undo changes if something went wrong
        print(f"Error occurred: {str(e)}")
        
    #rt.undo(False)  # End undo block

# Run the script
if __name__ == '__main__':
    main()