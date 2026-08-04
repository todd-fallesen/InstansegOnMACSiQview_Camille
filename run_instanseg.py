import sys
sys.path.insert(0, r"C:\Users\charoyc\AppData\Roaming\Python\Python39\site-packages")

import os
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

for mod_name in list(sys.modules.keys()):
    if "instanseg_gui" in mod_name:
        del sys.modules[mod_name]

from PySide2.QtWidgets import (QApplication, QDialog, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QScrollArea, QCheckBox, QWidget, QLineEdit, QProgressBar)
from PySide2.QtCore import Qt
import numpy as np
import tifffile
import xml.etree.ElementTree as ET
from instanseg import InstanSeg
import datetime
import time

print("SCRIPT STARTED")

DEFAULT_CHANNELS = ["DAPI", "CD45", "CD31", "F480", "Pan-Cytokeratin", "CD68", "SMA"]
KNOWN_CONTROLS = ["mouseIgGa", "mouseIgGa2", "mouseIgGb", "ratIgG", "Blocking", "Dk_anti", "Dk_antiGt"]
MODEL_NAME = "fluorescence_nuclei_and_cells"
TILE_SIZE = 512
OVERLAP = 64
BATCH_SIZE = 8

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

# -------------------------------------------------------
# Step 1: Select input - folder (MACSima) or file (OME-TIFF)
# -------------------------------------------------------
input_dialog = QDialog()
input_dialog.setWindowTitle("InstanSeg - Step 1: Select Input")
input_dialog.setMinimumWidth(500)
input_layout = QVBoxLayout()

input_label = QLabel("No input selected")
input_label.setWordWrap(True)
input_label.setProperty("path", None)
input_label.setProperty("mode", None)

def browse_macsima():
    folder = QFileDialog.getExistingDirectory(input_dialog, "Select MACSima ROI Folder")
    if folder:
        input_label.setText("MACSima folder: " + folder)
        input_label.setProperty("path", folder)
        input_label.setProperty("mode", "macsima")
        print("Selected MACSima folder:", folder)

def browse_ometiff():
    f, _ = QFileDialog.getOpenFileName(input_dialog, "Select OME-TIFF file", "", "TIFF files (*.tif *.tiff *.ome.tif *.ome.tiff)")
    if f:
        input_label.setText("OME-TIFF file: " + f)
        input_label.setProperty("path", f)
        input_label.setProperty("mode", "ometiff")
        print("Selected OME-TIFF:", f)

macsima_btn = QPushButton("Choose MACSima ROI Folder")
macsima_btn.clicked.connect(browse_macsima)
ometiff_btn = QPushButton("Choose Phenocycler OME-TIFF File")
ometiff_btn.clicked.connect(browse_ometiff)
next_btn = QPushButton("Next: Select Channels ->")
next_btn.clicked.connect(input_dialog.accept)
cancel_btn_1 = QPushButton("Cancel")
cancel_btn_1.clicked.connect(input_dialog.reject)

input_layout.addWidget(QLabel("Select your input data:"))
input_layout.addWidget(macsima_btn)
input_layout.addWidget(ometiff_btn)
input_layout.addWidget(input_label)
btn_row_1 = QHBoxLayout()
btn_row_1.addWidget(cancel_btn_1)
btn_row_1.addWidget(next_btn)
input_layout.addLayout(btn_row_1)
input_dialog.setLayout(input_layout)

result = input_dialog.exec_()
if result != QDialog.Accepted:
    print("Cancelled.")

input_path = input_label.property("path")
input_mode = input_label.property("mode")

if not input_path or not input_mode:
    print("No input selected, exiting.")

print("Mode:", input_mode)
print("Input:", input_path)

# -------------------------------------------------------
# Step 2: Scan channels based on mode
# -------------------------------------------------------
# channel_entries: list of dicts with keys:
#   key: unique key for this entry
#   label: display label for checkbox
#   default: whether pre-ticked
#   file: file path (MACSima) or None (OME-TIFF)
#   channel_index: index in stack (OME-TIFF) or None (MACSima)

channel_entries = []

if input_mode == "macsima":
    print("Scanning MACSima folder for channels...")
    def cycle_sort_key(folder_name):
        try:
            return int(folder_name.split("_")[0])
        except ValueError:
            return 9999
    cycle_folders = sorted([f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))], key=cycle_sort_key)

    dapi_found = False
    for cycle in cycle_folders:
        cycle_path = os.path.join(input_path, cycle)
        for fname in sorted(os.listdir(cycle_path)):
            if not fname.endswith(".tif") or "_A-" not in fname or "_B_" in fname:
                continue
            if "_S_" not in fname and "_AFS_" not in fname:
                continue

            antibody = fname.split("_A-")[-1].replace(".tif", "").split("_")[0]

            try:
                if "_S_" in fname:
                    fluorophore = fname.split("_S_")[1].split("_R-")[0]
                elif "_AFS_" in fname:
                    fluorophore = fname.split("_AFS_")[1].split("_R-")[0]
                else:
                    fluorophore = "unknown"
            except IndexError:
                fluorophore = "unknown"

            try:
                cycle_num = fname.split("_")[0]
            except IndexError:
                cycle_num = "?"

            if antibody == "DAPI":
                if dapi_found:
                    continue
                dapi_found = True

            is_control = any(ctrl.lower() in antibody.lower() for ctrl in KNOWN_CONTROLS)
            label = antibody + " (" + cycle_num + ", " + fluorophore + ")"
            if is_control:
                label += "  [control - not recommended]"

            channel_entries.append({
                "key": cycle + "__" + fname,
                "label": label,
                "antibody": antibody,
                "default": antibody in DEFAULT_CHANNELS and not is_control,
                "file": os.path.join(cycle_path, fname),
                "channel_index": None,
                "is_control": is_control
            })

    print("Found", len(channel_entries), "channel entries")
    print("Channels from 3_Scan2:")
    for e in channel_entries:
        if "3_Scan2" in e["key"]:
            print(" ", e["label"], "| default:", e["default"])

elif input_mode == "ometiff":
    print("Reading channel names from OME metadata...")
    with tifffile.TiffFile(input_path) as tif:
        ome_xml = tif.ome_metadata
        img_shape = tif.series[0].shape
        root = ET.fromstring(ome_xml)
        ns = root.tag.split("}")[0] + "}"
        pixels = root.find(".//" + ns + "Pixels")
        channels = pixels.findall(ns + "Channel")
        channel_names_raw = [ch.get("Name", "Channel_" + str(i)) for i, ch in enumerate(channels)]

    img_c, img_h, img_w = img_shape
    print("Image:", img_c, "channels,", img_h, "x", img_w, "pixels")

    from collections import Counter
    name_counts = Counter(channel_names_raw)

    name_seen = {}
    for i, name in enumerate(channel_names_raw):
        is_duplicate = name_counts[name] > 1
        if name not in name_seen:
            name_seen[name] = 1
        else:
            name_seen[name] += 1

        label = name
        if is_duplicate:
            label += " (occurrence " + str(name_seen[name]) + ")"
        channel_entries.append({
            "key": str(i) + "__" + name,
            "label": label,
            "antibody": name,
            "default": any(d.lower() in name.lower() for d in DEFAULT_CHANNELS) and not is_duplicate,
            "file": None,
            "channel_index": i,
            "is_control": False
        })

    print("Found", len(channel_entries), "channels")

# -------------------------------------------------------
# Step 3: Channel + settings dialog
# -------------------------------------------------------
channel_dialog = QDialog()
channel_dialog.setWindowTitle("InstanSeg - Step 2: Settings & Channels")
channel_dialog.setMinimumWidth(500)
channel_layout = QVBoxLayout()

widgets = {}

channel_layout.addWidget(QLabel("Pixel size (um):"))
widgets["pixel_size"] = QLineEdit("0.17" if input_mode == "macsima" else "0.5")
channel_layout.addWidget(widgets["pixel_size"])

channel_layout.addWidget(QLabel("Output folder:"))
widgets["output_label"] = QLabel("Default: saved next to input")
widgets["output_label"].setWordWrap(True)
widgets["output_label"].setProperty("path", None)

def browse_output():
    folder = QFileDialog.getExistingDirectory(channel_dialog, "Select Output Folder")
    if folder:
        widgets["output_label"].setText(folder)
        widgets["output_label"].setProperty("path", folder)
        print("Output folder:", folder)

browse_output_btn = QPushButton("Choose Output Folder")
browse_output_btn.clicked.connect(browse_output)
channel_layout.addWidget(browse_output_btn)
channel_layout.addWidget(widgets["output_label"])

channel_layout.addWidget(QLabel(" "))
if input_mode == "macsima":
    channel_layout.addWidget(QLabel("Select channels for segmentation:"))
    channel_layout.addWidget(QLabel("Format: Antibody (Cycle, Fluorophore). Controls unticked by default."))
else:
    channel_layout.addWidget(QLabel("Select channels for segmentation:"))
    channel_layout.addWidget(QLabel("Duplicate channels flagged by occurrence number."))

scroll_widget = QWidget()
scroll_layout = QVBoxLayout()
scroll_widget.setLayout(scroll_layout)

checkboxes = {}
for entry in channel_entries:
    cb = QCheckBox(entry["label"])
    cb.setChecked(entry["default"])
    scroll_layout.addWidget(cb)
    checkboxes[entry["key"]] = (cb, entry)

scroll = QScrollArea()
scroll.setWidget(scroll_widget)
scroll.setWidgetResizable(True)
scroll.setMinimumHeight(350)
channel_layout.addWidget(scroll)

def on_run():
    try:
        print("RUN BUTTON PRESSED")
        channel_dialog.accept()
    except Exception as e:
        print("ERROR IN on_run:", str(e))

run_btn = QPushButton("Run InstanSeg")
run_btn.clicked.connect(on_run)
cancel_btn_2 = QPushButton("Cancel")
cancel_btn_2.clicked.connect(channel_dialog.reject)

btn_row_2 = QHBoxLayout()
btn_row_2.addWidget(cancel_btn_2)
btn_row_2.addWidget(run_btn)
channel_layout.addLayout(btn_row_2)
channel_dialog.setLayout(channel_layout)

result = channel_dialog.exec_()
if result != QDialog.Accepted:
    print("Cancelled.")
else:
    try:
        pixel_size = float(widgets["pixel_size"].text())
    except ValueError:
        print("Invalid pixel size, using default 0.17")
        pixel_size = 0.17

    custom_output = widgets["output_label"].property("path")
    if custom_output:
        output_base = custom_output
    elif input_mode == "macsima":
        output_base = os.path.join(input_path, "instanseg_selected")
    else:
        output_base = os.path.join(os.path.dirname(input_path), os.path.basename(input_path).split(".")[0] + "_instanseg")

    selected_entries = [entry for key, (cb, entry) in checkboxes.items() if cb.isChecked()]
    print("Pixel size:", pixel_size, "um")
    print("Output folder:", output_base)
    print("Selected channels:", [e["label"] for e in selected_entries])

    # -------------------------------------------------------
    # Step 4: Open image data
    # -------------------------------------------------------
    print("Loading InstanSeg model...")
    model = InstanSeg(MODEL_NAME)
    model.inference_device = DEVICE
    model.instanseg = model.instanseg.to(DEVICE)
    print("Model loaded on:", DEVICE)

    use_memmap = False
    img_mmap = None
    ome_tif_handle = None
    channel_mmaps = []

    if input_mode == "macsima":
        channel_files = [e["file"] for e in selected_entries]
        print("Opening channel files...")
        for f in channel_files:
            channel_mmaps.append(tifffile.memmap(f, mode='r'))
        print("All channels opened.")
        with tifffile.TiffFile(channel_files[0]) as tif:
            page = tif.pages[0]
            img_h, img_w = page.shape[0], page.shape[1]

    else:
        channel_indices = [e["channel_index"] for e in selected_entries]
        print("Opening OME-TIFF...")
        with tifffile.TiffFile(input_path) as tif:
            is_compressed = tif.pages[0].compression.value != 1
        if not is_compressed:
            img_mmap = tifffile.memmap(input_path, mode='r')
            print("Uncompressed - using memory map, shape:", img_mmap.shape)
            use_memmap = True
        else:
            print("Compressed - will decompress tiles on demand")
            use_memmap = False
            ome_tif_handle = tifffile.TiffFile(input_path)
        img_h, img_w = img_shape[1], img_shape[2]

    print("Full image size:", img_h, "x", img_w)

    # -------------------------------------------------------
    # Step 5: Calculate tiles
    # -------------------------------------------------------
    tile_step = TILE_SIZE - OVERLAP
    n_tiles_y = max(1, int(np.ceil((img_h - OVERLAP) / tile_step)))
    n_tiles_x = max(1, int(np.ceil((img_w - OVERLAP) / tile_step)))
    n_tiles_total = n_tiles_y * n_tiles_x
    print("Processing", n_tiles_total, "tiles (" + str(n_tiles_y) + " rows x " + str(n_tiles_x) + " cols)")

    tile_coords = []
    for row in range(n_tiles_y):
        for col in range(n_tiles_x):
            y0 = row * tile_step
            x0 = col * tile_step
            y1 = min(y0 + TILE_SIZE, img_h)
            x1 = min(x0 + TILE_SIZE, img_w)
            y0 = max(0, y1 - TILE_SIZE)
            x0 = max(0, x1 - TILE_SIZE)
            core_y0 = y0 + (OVERLAP // 2) if row > 0 else y0
            core_x0 = x0 + (OVERLAP // 2) if col > 0 else x0
            core_y1 = y1 - (OVERLAP // 2) if row < n_tiles_y - 1 else y1
            core_x1 = x1 - (OVERLAP // 2) if col < n_tiles_x - 1 else x1
            tile_coords.append((y0, x0, y1, x1, core_y0, core_x0, core_y1, core_x1))

    # -------------------------------------------------------
    # Step 6: Progress dialog
    # -------------------------------------------------------
    progress_dialog = QDialog()
    progress_dialog.setWindowTitle("InstanSeg - Running Segmentation")
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setWindowFlags(progress_dialog.windowFlags() & ~Qt.WindowCloseButtonHint)
    progress_layout = QVBoxLayout()

    progress_title = QLabel("Running tiled segmentation...")
    progress_title.setStyleSheet("font-weight: bold;")
    progress_layout.addWidget(progress_title)

    mode_str = "MACSima" if input_mode == "macsima" else "Phenocycler OME-TIFF"
    progress_info = QLabel("Mode: " + mode_str + " | Image: " + str(img_w) + " x " + str(img_h) + " px | Channels: " + str(len(selected_entries)) + " | Tiles: " + str(n_tiles_total) + " | Device: " + DEVICE)
    progress_info.setWordWrap(True)
    progress_layout.addWidget(progress_info)

    progress_label = QLabel("Starting...")
    progress_layout.addWidget(progress_label)

    progress_bar = QProgressBar()
    progress_bar.setMinimum(0)
    progress_bar.setMaximum(n_tiles_total)
    progress_bar.setValue(0)
    progress_layout.addWidget(progress_bar)

    eta_label = QLabel("Estimated time remaining: calculating...")
    progress_layout.addWidget(eta_label)

    warning_label = QLabel("During processing MACSiQview will appear frozen. Please wait.")
    warning_label.setWordWrap(True)
    progress_layout.addWidget(warning_label)

    progress_dialog.setLayout(progress_layout)
    progress_dialog.show()
    progress_dialog.raise_()
    app.processEvents()

    # -------------------------------------------------------
    # Step 7: Tiled segmentation
    # -------------------------------------------------------
    nuclei_output = np.zeros((img_h, img_w), dtype=np.uint16)
    cells_output = np.zeros((img_h, img_w), dtype=np.uint16)
    nuclei_counter = 1
    cells_counter = 1
    tile_num = 0
    start_time = time.time()

    for batch_start in range(0, n_tiles_total, BATCH_SIZE):
        batch_coords = tile_coords[batch_start:batch_start + BATCH_SIZE]
        batch_tiles = []

        for (y0, x0, y1, x1, core_y0, core_x0, core_y1, core_x1) in batch_coords:
            tile_h = y1 - y0
            tile_w = x1 - x0
            tile_channels = []

            if input_mode == "macsima":
                for mmap in channel_mmaps:
                    actual_y1 = min(y1, mmap.shape[0])
                    actual_x1 = min(x1, mmap.shape[1])
                    tile = np.zeros((tile_h, tile_w), dtype=np.float32)
                    tile[:actual_y1-y0, :actual_x1-x0] = mmap[y0:actual_y1, x0:actual_x1].astype(np.float32)
                    tile_channels.append(tile)
            else:
                for ch_idx in channel_indices:
                    if use_memmap:
                        src = img_mmap[ch_idx, y0:y1, x0:x1].astype(np.float32)
                    else:
                        src = ome_tif_handle.pages[ch_idx].asarray()[y0:y1, x0:x1].astype(np.float32)
                    tile = np.zeros((tile_h, tile_w), dtype=np.float32)
                    tile[:src.shape[0], :src.shape[1]] = src
                    tile_channels.append(tile)

            batch_tiles.append(np.stack(tile_channels, axis=0))

        for i, (y0, x0, y1, x1, core_y0, core_x0, core_y1, core_x1) in enumerate(batch_coords):
            labeled_output, _ = model.eval_small_image(batch_tiles[i], pixel_size)
            nuclei_tile = labeled_output[0, 0].numpy().astype(np.uint16)
            cells_tile = labeled_output[0, 1].numpy().astype(np.uint16)

            local_core_y0 = core_y0 - y0
            local_core_x0 = core_x0 - x0
            local_core_y1 = local_core_y0 + (core_y1 - core_y0)
            local_core_x1 = local_core_x0 + (core_x1 - core_x0)

            nuclei_core = nuclei_tile[local_core_y0:local_core_y1, local_core_x0:local_core_x1].copy()
            cells_core = cells_tile[local_core_y0:local_core_y1, local_core_x0:local_core_x1].copy()

            if nuclei_core.max() > 0:
                mask = nuclei_core > 0
                nuclei_core[mask] = nuclei_core[mask] + nuclei_counter
                nuclei_counter += int(nuclei_core.max()) + 1

            if cells_core.max() > 0:
                mask = cells_core > 0
                cells_core[mask] = cells_core[mask] + cells_counter
                cells_counter += int(cells_core.max()) + 1

            nuclei_output[core_y0:core_y1, core_x0:core_x1] = nuclei_core
            cells_output[core_y0:core_y1, core_x0:core_x1] = cells_core
            tile_num += 1

        elapsed = time.time() - start_time
        avg_per_tile = elapsed / tile_num
        remaining = avg_per_tile * (n_tiles_total - tile_num)
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        progress_bar.setValue(tile_num)
        progress_label.setText("Tile " + str(tile_num) + " of " + str(n_tiles_total))
        eta_label.setText("Estimated time remaining: " + str(mins) + " min " + str(secs) + " sec")
        app.processEvents()

    # Close file handles
    if ome_tif_handle is not None:
        ome_tif_handle.close()

    # -------------------------------------------------------
    # Step 8: Save masks and log
    # -------------------------------------------------------
    progress_label.setText("Saving masks...")
    app.processEvents()

    os.makedirs(output_base, exist_ok=True)
    nuclei_path = os.path.join(output_base, "nuclei_mask.tif")
    cells_path = os.path.join(output_base, "cells_mask.tif")

    tifffile.imwrite(nuclei_path, nuclei_output)
    tifffile.imwrite(cells_path, cells_output)

    log_path = os.path.join(output_base, "segmentation_log.txt")
    with open(log_path, "w") as log:
        log.write("InstanSeg Segmentation Log\n")
        log.write("==========================\n")
        log.write("Date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        log.write("Mode: " + input_mode + "\n")
        log.write("Input: " + input_path + "\n")
        log.write("Model: " + MODEL_NAME + "\n")
        log.write("Pixel size: " + str(pixel_size) + " um\n")
        log.write("Tile size: " + str(TILE_SIZE) + " px\n")
        log.write("Overlap: " + str(OVERLAP) + " px\n")
        log.write("Device: " + DEVICE + "\n")
        log.write("Image size: " + str(img_w) + " x " + str(img_h) + " px\n")
        log.write("Channels used (" + str(len(selected_entries)) + "):\n")
        for e in selected_entries:
            log.write("  " + e["label"] + "\n")
        log.write("Output:\n")
        log.write("  " + nuclei_path + "\n")
        log.write("  " + cells_path + "\n")
    print("Log saved:", log_path)

    progress_dialog.close()

    # -------------------------------------------------------
    # Step 9: Done dialog
    # -------------------------------------------------------
    done_dialog = QDialog()
    done_dialog.setWindowTitle("InstanSeg - Done!")
    done_layout = QVBoxLayout()
    done_layout.addWidget(QLabel("Segmentation complete!"))
    done_layout.addWidget(QLabel(" "))
    done_layout.addWidget(QLabel("Masks saved to:"))
    nuclei_lbl = QLabel(nuclei_path)
    nuclei_lbl.setWordWrap(True)
    cells_lbl = QLabel(cells_path)
    cells_lbl.setWordWrap(True)
    done_layout.addWidget(nuclei_lbl)
    done_layout.addWidget(cells_lbl)
    done_layout.addWidget(QLabel(" "))
    done_layout.addWidget(QLabel("Import via MACSiQview:\nSegmentation panel > Import Segmentation Mask"))
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(done_dialog.accept)
    done_layout.addWidget(close_btn)
    done_dialog.setLayout(done_layout)
    done_dialog.exec_()

    print("Done! Mask files saved:")
    print("  Nuclei:", nuclei_path)
    print("  Cells:", cells_path)