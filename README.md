# onbadexporter

A 3ds Max plugin for exporting Novalogic's animation format (.bad)

## Credits

Special thanks to the following contributors for their help and wisdom:

* DareHacker (DH)
* Oscarmike247

## Installation

1. Download or clone the repository
2. Open 3ds Max
3. Right-click in the mini listener window and select "Open Editor" 
4. Navigate to File -> Open and select `onbadexporter.py`
5. Press Ctrl+E to run the plugin

## Example Usage

The repository includes an example AK47 animation with the following files:

* A .max file containing the model
* A .3di file (rexported stock model for the updated rig)
* A configuration file

### Testing the Example Animation

1. Launch the plugin following the installation steps above
2. Load the provided configuration file
3. Update the export directory to point to your game directory
4. Press export (Note: This will overwrite existing files in the export directory)
5. Add `/d` to your game's launch options
6. Copy the provided .3di file to your game directory
7. Test the animation in-game

**Note:** Some weapon.def parameters may need adjustment for optimal appearance.

**Additional Information:** The .max file can also be exported to .ase format
and imported into SuperOED if needed. This would allow updating the model too.
