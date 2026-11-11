# MACSiQview

# This is originally Camille's code. Camille is on maternity leave and I forgot to click her invitiation within in a week to colloborate.  This code will eventually merge back in to her original repo.

Scripts needed to install and run Instanseg on MACSiQview for cell segmentation of images generated on the MACSima or on the PhenoCycler

Copy files in C:\Users\username\Documents\QiScripts\instanseg

In MACSiQview in the python menu go in instanseg and run to install the packages necessary to run instanseg (these only need to be run once):
- 1_install_instanseg.py
- 2_patch_instanseg_ChangeFilePath.py - Change file path to your username before running script
- 3_reinstall_pytorch_ChangeFilePath.py - Change file path to your username before running script
- 4_cudaGPU_instal.py
during install MACSiQview will appear frozen, that should only last a few minutes
the python console will update and confirm the installation after the script has finished running

To run Instanseg, just go in the python menu. In the Instanseg folder click on run_instanseg.

A window will open asking you to select the folder where the images are if loading data from a MACSima experiemnt or to select the image if loading data from a PhenoCycler experiment.

After clicking on next, you will be able to select the channels you want to use to run the segmentation. The more channels selected the longer it will take to run (in our experience using all the channels gives a poorer segmentation result than using a selected few...)


Good luck with your segmentation!
