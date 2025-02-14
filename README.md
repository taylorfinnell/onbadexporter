# onbadexporter

A 3ds Max plugin for exporting Novalogic's animation format (.bad)

## Credits

Special thanks to the following contributors

* DareHacker (DH)
* Oscarmike247
*  @mohammed_furqon

## Installation

1. Download or clone the repository
2. Open 3ds Max
3. Right-click in the mini listener window and select "Open Editor"
4. Navigate to File -> Open and select `onbadexporter.py`
5. Press Ctrl+E to run the plugin

## Example

The repository includes an example AK47 animation with the following files:

* A .max file containing the model
* A configuration file
* The pre-generated .bad files (if you don't want to generate them yourself)

If you want to generate the files .bad files yourself or modify the animation(s) you
can do the following

1. Launch the plugin following the installation steps above
2. Load the provided configuration file
3. Update the export directory to point to your game directory
4. Press export (Note: This will overwrite existing files in the export directory)
5. Add `/d` to your game's launch options
7. Test the animation in-game

**TODO:** Some weapon.def parameters may need adjustment for optimal appearance. There is an animation delay in the weapon.def that makes the switchto animation not properly display.

**NOTE:** This particular rig only works with the stock model since it created by imported a .bad file. So it's ~NovaLogic's original rig. The has a hardcoded translation that only works on the stock model. Note that if you use your own rig you will need to export a fresh model of the ak47 to capture the bindpose of your setup.
