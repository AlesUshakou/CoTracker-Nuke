#!/usr/bin/env python3
"""
Gradio UI Interface (with --video autoload + proper Process by state)
"""
import gradio as gr
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Any
import os, sys
from pathlib import Path
from datetime import datetime
import time

from ..core.app import CoTrackerNukeApp
from ..exporters.stmap_exporter import STMapExporter


class GradioInterface:
    def __init__(self, app: CoTrackerNukeApp):
        self.app = app
        self.logger = app.logger
        self.preview_video_path = None
        self.last_exported_path = None
        self.last_stmap_path = None

    # ---------- Video load & helpers ----------
    def load_video_for_reference(self, reference_video, start_frame_offset) -> Tuple[str, Optional[str], dict, dict, dict, Optional[str]]:
        """
        Load video and return:
           status, video_path_for_player, frame_slider_update, stmap_start_update, stmap_end_update, current_video_path_state
        """
        try:
            if reference_video is None:
                return "❌ No video file selected", None, gr.update(), gr.update(), gr.update(), None
            self.app.load_video(reference_video)
            self.preview_video_path = reference_video

            info = self.app.get_video_info()
            fps_info = self.get_video_fps(reference_video)

            status_msg = (f"✅ Video loaded successfully!\n"
                          f"📹 Frames: {info['frames']}\n"
                          f"📐 Resolution: {info['width']}x{info['height']}\n"
                          f"🎬 FPS: {fps_info}\n"
                          f"💾 Size: {info['memory_mb']:.1f} MB")

            start_offset = start_frame_offset if start_frame_offset is not None else 1001
            max_frame = start_offset + info['frames'] - 1
            slider_update = gr.update(minimum=start_offset, maximum=max_frame, value=start_offset)
            return status_msg, reference_video, slider_update, gr.update(value=start_offset), gr.update(value=max_frame), reference_video
        except Exception as e:
            self.logger.error(f"Error loading video: {e}")
            return f"❌ Error loading video: {e}", None, gr.update(), gr.update(), gr.update(), None

    def update_frame_slider_range(self, reference_video, start_frame_offset) -> dict:
        try:
            if reference_video is None or self.app.current_video is None:
                return gr.update()
            info = self.app.get_video_info()
            start_offset = start_frame_offset if start_frame_offset is not None else 1001
            max_frame = start_offset + info['frames'] - 1
            return gr.update(minimum=start_offset, maximum=max_frame, value=start_offset)
        except Exception as e:
            self.logger.error(f"Error updating slider: {e}")
            return gr.update()

    def set_manual_reference_frame(self, frame_number_with_offset: int, start_frame_offset: int) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        try:
            if frame_number_with_offset < start_frame_offset:
                return None, None
            frame_number = frame_number_with_offset - start_frame_offset
            actual = self.app.set_reference_frame(frame_number)
            frame_image = self.app.get_reference_frame_image()
            if frame_image is None:
                return None, None
            pil = Image.fromarray(frame_image.astype(np.uint8))
            self.logger.info(f"Reference frame set to {frame_number_with_offset} (internal {actual})")
            return pil, pil
        except Exception as e:
            self.logger.error(f"Error setting ref frame: {e}")
            return None, None

    def update_frame_from_input(self, frame_number_with_offset: int, start_frame_offset: int) -> Optional[Image.Image]:
        try:
            if self.app.current_video is None:
                return None
            if frame_number_with_offset < start_frame_offset:
                return None
            frame_number = frame_number_with_offset - start_frame_offset
            frame = self.app.video_processor.get_frame(int(frame_number))
            if frame is None:
                return None
            return Image.fromarray(frame.astype(np.uint8))
        except Exception as e:
            self.logger.error(f"Error displaying frame: {e}")
            return None

    def calculate_grid_info(self, grid_size: int) -> dict:
        try:
            if self.app.video_processor.current_video is not None:
                h, w = self.app.video_processor.current_video.shape[1:3]
                if w >= h:
                    gw = grid_size; gh = max(1, int(round(grid_size * h / w)))
                else:
                    gh = grid_size; gw = max(1, int(round(grid_size * w / h)))
                total = gw * gh
                has_mask = self.app.mask_handler.current_mask is not None
                if has_mask:
                    mask = self.app.mask_handler.current_mask
                    white = int(np.sum(mask == 255))
                    cov = white / (mask.shape[0] * mask.shape[1])
                    eff = int(total * cov)
                    s = f"📊 Grid: {gw}×{gh}={total:,} | With mask ≈{eff:,} ({cov*100:.1f}%)"
                    if eff > 300: s += f"\n⚠️ High VRAM usage: {eff:,} points"
                else:
                    s = f"📊 Grid: {gw}×{gh}={total:,} (no mask)"
                    if total > 300: s += f"\n⚠️ High VRAM usage: {total:,} points"
                return gr.update(value=s, visible=True)
            return gr.update(value="⚠️ Load video first", visible=True)
        except Exception as e:
            self.logger.error(f"Grid info error: {e}")
            return gr.update(value="❌ Error calculating points", visible=True)

    # ---------- Processing using STATE (fixes 'No video loaded') ----------
    def process_video_by_path(self, video_path: Optional[str], grid_size: int, image_sequence_start_frame: int = 1001):
        try:
            if not video_path or not os.path.exists(video_path):
                return "❌ No video loaded", None
            # Ensure loaded
            if self.app.current_video is None or self.preview_video_path != video_path:
                self.app.load_video(video_path)
                self.preview_video_path = video_path

            _tracks, _vis = self.app.track_points(grid_size)
            preview_video_path = self.app.create_preview_video(frame_offset=image_sequence_start_frame)

            info = self.app.get_tracking_info()
            ref_disp = (self.app.reference_frame or 0) + image_sequence_start_frame
            mask_status = "✅ Used" if self.app.mask_handler.current_mask is not None else "❌ None"
            msg = (f"✅ Tracking completed!\n"
                   f"🎯 Points tracked: {info['num_points']}\n"
                   f"📹 Frames: {info['num_frames']}\n"
                   f"🎬 Reference frame: {ref_disp}\n"
                   f"🎭 Mask: {mask_status}\n"
                   f"👁️ Visibility: {info['visibility_rate']:.1f}%\n"
                   f"📊 Total detections: {info['total_detections']}/{info['possible_detections']}\n"
                   f"🎬 Preview: {'Created successfully' if preview_video_path else 'Failed to create'}")
            return msg, preview_video_path
        except Exception as e:
            self.logger.error(f"Process error: {e}")
            return f"❌ Error processing video: {e}", None

    def use_mask_from_editor(self, edited_image: Any) -> str:
        try:
            if edited_image is None:
                return "❌ No mask drawn. Draw a white mask on the reference frame."
            message, _ = self.app.process_mask_from_editor(edited_image)
            return message
        except Exception as e:
            self.logger.error(f"Mask error: {e}")
            return f"❌ Error processing mask: {e}"

    # ---------- Export helpers ----------
    def get_default_output_path(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("outputs"); out.mkdir(exist_ok=True)
        return str((out / f"CoTracker_{ts}.nk").resolve())

    def get_default_stmap_output_path(self, reference_frame: int = None) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("outputs"); out.mkdir(exist_ok=True)
        ref = str(reference_frame) if reference_frame is not None else "%refFrame%"
        return str((out / f"CoTracker_{ts}_stmap_ref{ref}/CoTracker_{ts}_stmap_ref{ref}.%04d.exr").resolve())

    def process_path_variables(self, path_template: str, reference_frame: int = None) -> str:
        return path_template.replace("%refFrame%", str(reference_frame)) if reference_frame is not None else path_template

    def browse_output_folder(self) -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', 1)
            cur = self.get_default_output_path()
            curdir = os.path.dirname(os.path.abspath(cur))
            os.makedirs(curdir, exist_ok=True)
            fp = filedialog.asksaveasfilename(title="Save Nuke file as.",
                                              initialdir=curdir,
                                              defaultextension=".nk",
                                              filetypes=[("Nuke files","*.nk"),("All files","*.*")],
                                              initialfile=os.path.basename(cur))
            root.destroy()
            if fp:
                fp = fp.replace('\\','/')
                try: return os.path.relpath(fp).replace('\\','/')
                except ValueError: return fp
            return self.get_default_output_path()
        except Exception:
            return self.get_default_output_path()

    def browse_stmap_output_folder(self) -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', 1)
            outdir = os.path.abspath("outputs"); os.makedirs(outdir, exist_ok=True)
            fp = filedialog.asksaveasfilename(title="Save STMap sequence as.",
                                              initialdir=outdir,
                                              defaultextension=".exr",
                                              filetypes=[("EXR files","*.exr"),("All files","*.*")],
                                              initialfile="CoTracker_stmap_ref%refFrame%.%04d.exr")
            root.destroy()
            if fp: return fp.replace('\\','/')
            return self.get_default_stmap_output_path()
        except Exception:
            return self.get_default_stmap_output_path()

    def export_nuke_file(self, output_file_path, frame_offset: int) -> str:
        try:
            if not output_file_path:
                output_path = self.get_default_output_path()
            elif isinstance(output_file_path, str):
                output_path = output_file_path
            else:
                output_path = str(output_file_path)
            if self.app.tracking_results is None:
                return "❌ No tracking data available. Please process video first."
            if not output_path.endswith(".nk"): output_path += ".nk"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            nuke_path = self.app.export_to_nuke(output_path, frame_offset)
            self.last_exported_path = nuke_path
            info = self.app.get_tracking_info()
            return (f"✅ Export completed!\n"
                    f"📁 File: {nuke_path}\n"
                    f"🎯 Points: {info['num_points']}\n"
                    f"📹 Frames: {info['num_frames']}\n"
                    f"🔢 Frame offset: {frame_offset}\n"
                    f"📂 Directory: {Path(nuke_path).parent}")
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            return f"❌ Export failed: {e}"

    def copy_exported_path(self) -> str:
        if self.last_exported_path is None:
            return "❌ No file has been exported yet. Please export a .nk file first."
        return (f"📋 Copied to clipboard!\n{self.last_exported_path}"
                if self.copy_to_clipboard(self.last_exported_path)
                else f"⚠️ Could not copy to clipboard.\nPath: {self.last_exported_path}")

    def export_stmap_sequence(self, interpolation_method: str, bit_depth: int,
                              frame_start: int, frame_end: Optional[int],
                              image_sequence_start_frame: int = 1001,
                              output_file_path: Optional[str] = None,
                              progress=gr.Progress()) -> str:
        try:
            if output_file_path in (None, ""):
                output_file_path = self.get_default_stmap_output_path(self.app.reference_frame + image_sequence_start_frame if self.app.reference_frame is not None else None)
            processed_path = self.process_path_variables(output_file_path, self.app.reference_frame + image_sequence_start_frame if self.app.reference_frame is not None else None)
            Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
            exporter = STMapExporter(self.app)
            tracks, visibility = self.app.get_tracks_and_visibility()
            mask = self.app.mask_handler.current_mask
            if frame_end is None:
                info = self.app.get_video_info()
                frame_end = (self.app.reference_frame + image_sequence_start_frame) + info['frames'] - 1
            frame_start = frame_start if frame_start is not None else (self.app.reference_frame + image_sequence_start_frame)
            processing_time_seconds = None
            def progress_callback_simple(current_frame, total_frames, message=None):
                nonlocal processing_time_seconds
                if message and "seconds" in message:
                    import re
                    m = re.search(r'(\d+\.?\d*)\s*seconds', message)
                    if m: processing_time_seconds = float(m.group(1))
                progress(0, desc=message or f"{current_frame}/{total_frames}")
            output_dir = exporter.generate_stmap_sequence(tracks=tracks, visibility=visibility, mask=mask,
                                                          interpolation_method=interpolation_method, bit_depth=bit_depth,
                                                          frame_start=frame_start, frame_end=frame_end,
                                                          frame_offset=image_sequence_start_frame,
                                                          output_file_path=processed_path,
                                                          progress_callback=progress_callback_simple)
            self.last_stmap_path = str(Path(output_dir).resolve())
            exrs = list(Path(output_dir).glob("*.exr"))
            timing = (lambda s: f"{int(s//60):02d}min {int(s%60):02d}sec")(processing_time_seconds) if processing_time_seconds else "XXmin XXsec"
            return (f"✅ STMap sequence generated in {timing}\n"
                    f"📁 Directory: {self.last_stmap_path}\n"
                    f"📹 Frames: {len(exrs)} RGBA EXR files\n"
                    f"🎬 Reference frame: {self.app.reference_frame + image_sequence_start_frame}\n"
                    f"🎯 Features: Mask-aware interpolation, RGBA output")
        except Exception as e:
            self.logger.error(f"STMap export error: {e}")
            return f"❌ STMap export failed: {e}"


    def copy_stmap_directory_path(self) -> str:
        """
        Copy the last exported STMap directory path to clipboard.
        Uses self.last_stmap_path which is set after successful STMap export.
        """
        try:
            if self.last_stmap_path is None:
                return "❌ No STMap sequence has been exported yet. Please export STMap first."
            ok = self.copy_to_clipboard(self.last_stmap_path)
            if ok:
                return f"📋 Copied to clipboard!\n{self.last_stmap_path}"
            else:
                return f"⚠️ Could not copy to clipboard.\nPath: {self.last_stmap_path}"
        except Exception as e:
            self.logger.error(f"Copy STMap dir path error: {e}")
            return f"❌ Copy failed: {e}"

    # ---- Misc helpers ----
    def copy_to_clipboard(self, text: str) -> bool:
        try:
            try:
                import pyperclip
                pyperclip.copy(text); return True
            except Exception:
                pass
            if sys.platform.startswith("win"):
                import subprocess
                p = subprocess.Popen(['clip'], stdin=subprocess.PIPE, close_fds=True)
                p.communicate(input=text.encode('utf-8')); return p.returncode == 0
            if sys.platform == "darwin":
                import subprocess
                subprocess.run(['pbcopy'], input=text, text=True, check=True, timeout=5); return True
            if sys.platform.startswith("linux"):
                import subprocess
                try:
                    subprocess.run(['xclip','-selection','clipboard'], input=text, text=True, check=True, timeout=5); return True
                except Exception:
                    subprocess.run(['xsel','--clipboard','--input'], input=text, text=True, check=True, timeout=5); return True
        except Exception:
            return False
        return False

    def update_stmap_frame_defaults(self, reference_video, image_sequence_start_frame) -> Tuple[Optional[int], Optional[int]]:
        try:
            if reference_video is None or self.app.current_video is None:
                return gr.update(), gr.update()
            info = self.app.get_video_info()
            start_offset = image_sequence_start_frame if image_sequence_start_frame is not None else 1001
            max_frame = start_offset + info['frames'] - 1
            return gr.update(value=start_offset), gr.update(value=max_frame)
        except Exception as e:
            self.logger.error(f"STMap defaults error: {e}")
            return gr.update(), gr.update()

    def get_video_fps(self, video_path: str) -> str:
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
            return f"{fps:.2f}" if fps > 0 else "Unknown"
        except Exception:
            return "Unknown"

    # ---------- Build UI ----------
    def create_interface(self, prefill_video: Optional[str] = None) -> gr.Blocks:
        with gr.Blocks(
            title="CoTracker Nuke Integration",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {max-width: 1200px; margin: auto; width: 100%;}
            .yellow-button {background-color: #ff8c00 !important; color: #000 !important; border: 2px solid #e67e00 !important;}
            .yellow-button:hover {background-color: #e67e00 !important; color: #000 !important;}
            .green-button {background-color: #2e7d32 !important; color: #fff !important; border: 2px solid #1b5e20 !important;}
            .green-button:hover {background-color: #1b5e20 !important; color: #fff !important;}
            """
        ) as interface:
            gr.Markdown("# 🎬 CoTracker Nuke Integration")
            gr.Markdown("Track points in video using CoTracker and export to Nuke.")

            # Step 1
            gr.Markdown("## 📹 Step 1: Upload Video")
            reference_video = gr.File(label="📁 Upload Video File",
                                      file_types=[".mp4",".mov",".avi",".mkv",".webm"],
                                      type="filepath")
            video_player = gr.Video(label="📹 Video Player", height=300)
            video_status = gr.Textbox(label="📊 Video Status", interactive=False, lines=4)

            # keep current path in STATE
            current_video_path = gr.State(value=None)

            # Step 2
            gr.Markdown("## 🎬 Step 2: Set Image Sequence Start Frame")
            image_sequence_start_frame = gr.Number(label="🎬 Image Sequence Start Frame",
                                                   value=1001,
                                                   info="Frame number where your image sequence starts in Nuke")

            # Step 3
            gr.Markdown("## 🎯 Step 3: Set Reference Frame")
            with gr.Row():
                with gr.Column(scale=2):
                    frame_display = gr.Image(label="🖼️ Reference Frame Preview", height=300, type="pil")
                with gr.Column(scale=1):
                    frame_slider = gr.Slider(minimum=1001, maximum=1100, step=1, value=1001,
                                             label="🎬 Frame #",
                                             info="Frame number for tracking reference (includes start frame offset)")
                    set_manual_frame_btn = gr.Button("📤 Set Reference Frame", variant="primary", size="lg")

            # Step 4
            gr.Markdown("## 🎨 Step 4: Optional Mask Drawing")
            with gr.Row():
                with gr.Column(scale=3):
                    mask_editor = gr.ImageEditor(label="🖼️ Reference Frame - Draw Mask",
                                                 type="pil",
                                                 brush=gr.Brush(colors=["#FFFFFF","#000000"], default_size=20),
                                                 height=400, interactive=True)
                with gr.Column(scale=1):
                    use_mask_btn = gr.Button("🎯 Use/Update Mask", variant="primary", size="lg")
                    mask_result = gr.Textbox(label="✅ Mask Status", interactive=False, lines=4)

            # Step 5
            gr.Markdown("## 🚀 Step 5: Process Video")
            with gr.Row():
                with gr.Column(scale=2):
                    grid_size = gr.Slider(minimum=5, maximum=400, step=1, value=40,
                                          label="🔢 Grid Size (Points on Longest Side)",
                                          info="Higher values = more tracking points")
                    vram_warning = gr.Textbox(label="⚠️ VRAM Warning", interactive=False, lines=2, visible=False)
                with gr.Column(scale=1):
                    process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
            processing_status = gr.Textbox(label="⚙️ Processing Status", interactive=False, lines=4)
            gr.Markdown("### 🎬 Tracking Results")
            preview_video = gr.Video(label="📹 Tracking Preview", height=400)

            # Step 6
            gr.Markdown("## 📤 Step 6: Export to Nuke")
            with gr.Group():
                with gr.Row():
                    gr.Markdown("**.nk Output File Path**", elem_classes="yellow-label")
                    file_picker_btn = gr.Button("📂 Browse", size="sm", scale=1, elem_classes="yellow-button")
                output_file_path = gr.Textbox(value=self.get_default_output_path(),
                                              info="Path where the .nk file will be saved",
                                              show_label=False)
            export_btn = gr.Button("📤 Generate Tracker Node as .nk", variant="primary", size="lg", elem_classes="yellow-button")
            export_status = gr.Textbox(label="📋 Export Status", interactive=False, lines=4)
            copy_path_btn = gr.Button("📋 Copy .nk Path to Clipboard", variant="primary", size="lg", elem_classes="yellow-button")
            copy_status = gr.Textbox(label="📋 Copy Status", interactive=False, lines=2)

            # STMap
            gr.Markdown("## 🗺️ Step 7: Export STMap Sequence")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        stmap_interpolation = gr.Dropdown(choices=["linear","cubic"], value="linear",
                                                          label="🔧 Interpolation", scale=1)
                        stmap_bit_depth = gr.Dropdown(choices=[16,32], value=32, label="💾 Bit Depth", scale=1)
                with gr.Column(scale=1):
                    with gr.Row():
                        stmap_frame_start = gr.Number(value=None, label="🎬 Start Frame", scale=1)
                        stmap_frame_end = gr.Number(value=None, label="🎬 End Frame", scale=1)
            with gr.Group():
                with gr.Row():
                    gr.Markdown("**STMap Output File Path**", elem_classes="green-label")
                    stmap_file_picker_btn = gr.Button("📂 Browse", size="sm", scale=1, elem_classes="green-button")
                stmap_output_file_path = gr.Textbox(value=self.get_default_stmap_output_path(),
                                                    info="Use %04d for frame numbers, %refFrame% for reference frame",
                                                    show_label=False)
            stmap_export_btn = gr.Button("🗺️ Generate STMap Sequence", variant="primary", size="lg", elem_classes="green-button")
            stmap_export_status = gr.Textbox(label="📋 STMap Export Status", interactive=False, lines=4)
            stmap_copy_path_btn = gr.Button("📋 Copy STMap Directory Path", variant="primary", size="lg", elem_classes="green-button")
            stmap_copy_status = gr.Textbox(label="📋 STMap Copy Status", interactive=False, lines=2)

            # ---------- Events ----------
            # when user selects a file manually: also update STATE with path
            reference_video.change(fn=self.load_video_for_reference,
                                   inputs=[reference_video, image_sequence_start_frame],
                                   outputs=[video_status, video_player, frame_slider, stmap_frame_start, stmap_frame_end, current_video_path])

            image_sequence_start_frame.change(fn=self.update_frame_slider_range,
                                              inputs=[current_video_path, image_sequence_start_frame],
                                              outputs=[frame_slider])

            set_manual_frame_btn.click(fn=self.set_manual_reference_frame,
                                       inputs=[frame_slider, image_sequence_start_frame],
                                       outputs=[frame_display, mask_editor])

            frame_slider.release(fn=self.update_frame_from_input,
                                 inputs=[frame_slider, image_sequence_start_frame],
                                 outputs=[frame_display])

            grid_size.release(fn=self.calculate_grid_info, inputs=[grid_size], outputs=[vram_warning])

            # Process now uses STATE (current_video_path) instead of gr.File
            process_btn.click(fn=self.process_video_by_path,
                              inputs=[current_video_path, grid_size, image_sequence_start_frame],
                              outputs=[processing_status, preview_video])

            use_mask_btn.click(fn=lambda edited, g: (self.use_mask_from_editor(edited), self.calculate_grid_info(g)),
                               inputs=[mask_editor, grid_size],
                               outputs=[mask_result, vram_warning])

            file_picker_btn.click(fn=lambda: gr.update(value=self.browse_output_folder()), outputs=[output_file_path])
            export_btn.click(fn=self.export_nuke_file, inputs=[output_file_path, image_sequence_start_frame], outputs=[export_status])
            copy_path_btn.click(fn=self.copy_exported_path, outputs=[copy_status])

            reference_video.change(fn=self.update_stmap_frame_defaults,
                                   inputs=[current_video_path, image_sequence_start_frame],
                                   outputs=[stmap_frame_start, stmap_frame_end])

            image_sequence_start_frame.change(fn=self.update_stmap_frame_defaults,
                                              inputs=[current_video_path, image_sequence_start_frame],
                                              outputs=[stmap_frame_start, stmap_frame_end])

            stmap_export_btn.click(fn=self.export_stmap_sequence,
                                   inputs=[stmap_interpolation, stmap_bit_depth, stmap_frame_start, stmap_frame_end, image_sequence_start_frame, stmap_output_file_path],
                                   outputs=[stmap_export_status])
            stmap_file_picker_btn.click(fn=self.browse_stmap_output_folder, outputs=[stmap_output_file_path])
            stmap_copy_path_btn.click(fn=self.copy_stmap_directory_path, outputs=[stmap_copy_status])

            # ===== Autoload on start -> also set STATE =====
            _prefill_path = prefill_video or os.getenv("COTRACKER_VIDEO")
            prefill_state = gr.State(value=_prefill_path)

            def _autorun_on_load(path_str, start_frame_offset):
                if not path_str:
                    return gr.update(), None, gr.update(), gr.update(), gr.update(), None
                p = os.path.abspath(str(path_str))
                if not os.path.exists(p):
                    return (f"❌ Specified --video not found: `{p}`", None, gr.update(), gr.update(), gr.update(), None)
                # will return state as last output
                return self.load_video_for_reference(p, start_frame_offset)

            interface.load(fn=_autorun_on_load,
                           inputs=[prefill_state, image_sequence_start_frame],
                           outputs=[video_status, video_player, frame_slider, stmap_frame_start, stmap_frame_end, current_video_path])

        return interface


def create_gradio_interface(debug_mode: bool = True, console_log_level: str = "INFO", prefill_video: Optional[str] = None) -> gr.Blocks:
    app = CoTrackerNukeApp(debug_mode, console_log_level)
    ui = GradioInterface(app)
    return ui.create_interface(prefill_video=prefill_video)
