import pymxs
import random

rt = pymxs.runtime

def reset_materials():
    """
    Removes all materials from scene geometry and applies a new diffuse material
    with a unique random color for each object.
    """
    all_objects = rt.objects
    modified_count = 0
    
    # Iterate through all objects in the scene
    for obj in all_objects:
        try:
            # Check if the object can have a material assigned
            if hasattr(obj, 'material'):
                # Generate a random RGB color (values from 0 to 255)
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                
                # Create a new standard material with the random diffuse color
                new_mat = rt.StandardMaterial()
                new_mat.diffuse = rt.Color(r, g, b)
                new_mat.name = f"SimpleDiffuse_{modified_count}"  # Unique material name
                
                # Apply the new material to the object
                obj.material = new_mat
                modified_count += 1
        except Exception as e:
            print(f"Failed to modify {obj.name}: {str(e)}")
    
    print(f"Modified {modified_count} objects with unique new materials")

def main():
    try:
        # Uncomment the following line if you want to create an undo point before changes
        # rt.undo(True)
        
        reset_materials()
        
        # Refresh the viewport to see the changes
        rt.redrawViews()
        print("Material reset completed successfully")
        
    except Exception as e:
        # Uncomment the following line if you want to undo the changes on error
        # rt.undo(False)
        print(f"Error occurred: {str(e)}")
        
    # Uncomment the following line to end the undo block if using undo
    # rt.undo(False)

# Run the script
if __name__ == '__main__':
    main()
