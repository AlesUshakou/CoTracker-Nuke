#!/usr/bin/env python
"""
CoTracker CSV to Nuke .nk File Generator
Direct .nk file creation based on analysis of working tracker
"""

import csv
import os
from datetime import datetime

def generate_nuke_file(csv_path, output_path=None, image_height=1080, min_confidence=0.5, frame_offset=0, reference_frame=0):
    """
    Generate a complete .nk file with Tracker4 node from CoTracker CSV data.
    
    Args:
        csv_path: Path to CoTracker CSV file
        output_path: Output .nk file path (auto-generated if None)
        image_height: Image height for coordinate conversion (default 1080)
        min_confidence: Minimum confidence threshold (default 0.5)
        frame_offset: Frame offset to apply to frame numbers (default 0)
        reference_frame: Reference frame number for the tracker (default 0)
    """
    
    # Auto-generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"cotracker_{base_name}_{timestamp}.nk"
    
    print(f"CoTracker CSV to Nuke .nk Generator")
    print(f"=" * 50)
    print(f"Input CSV: {csv_path}")
    print(f"Output .nk: {output_path}")
    print(f"Image Height: {image_height}")
    print(f"Min Confidence: {min_confidence}")
    print()
    
    # Read and process CSV data
    tracker_dict = {}
    total_rows = 0
    filtered_rows = 0
    
    print("Reading CSV data...")
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 6:
                continue
                
            total_rows += 1
            frame = int(row[0])
            point_id = int(row[1])
            x = float(row[2])
            y = float(row[3])
            visible = row[4].lower() == 'true'
            confidence = float(row[5])
            
            # Initialize point if not exists
            if point_id not in tracker_dict:
                tracker_dict[point_id] = []
            
            # Filter based on visibility and confidence
            if not visible or confidence < min_confidence:
                filtered_rows += 1
                continue
            
            # Convert Y coordinate: Nuke uses bottom-left origin, CoTracker uses top-left
            y_nuke = image_height - y
            
            # Apply frame offset and store valid keyframe
            frame_with_offset = frame + frame_offset
            tracker_dict[point_id].append([frame_with_offset, x, y_nuke])
    
    print(f"CSV Processing Complete:")
    print(f"  Total rows: {total_rows}")
    print(f"  Filtered out: {filtered_rows} (invisible or confidence < {min_confidence})")
    print(f"  Valid keyframes: {total_rows - filtered_rows}")
    print(f"  Tracking points: {len(tracker_dict)}")
    print()
    
    # Generate .nk file content
    print("Generating .nk file...")
    
    # Sort point IDs for consistent ordering
    point_ids = sorted(tracker_dict.keys())
    num_tracks = len(point_ids)
    
    # Build track data strings
    track_data_lines = []
    
    for i, point_id in enumerate(point_ids):
        keyframes = tracker_dict[point_id]
        
        # Build X and Y coordinate curves
        x_values = []
        y_values = []
        
        for frame, x, y in keyframes:
            x_values.append(str(x))
            y_values.append(str(y))
        
        x_curve = " ".join(x_values)
        y_curve = " ".join(y_values)
        
        frame_start = reference_frame
        name = f'track {i+1}'
        x_str = x_curve
        y_str = y_curve
        trs_val = "1 0 0"


        # Create track line (based on analysis of working .nk file)
        track_line =  ''' { {curve K x%s 1} "%s" {curve x%s %s} {curve x%s %s} 
                            {curve K x%s 0} {curve K x%s 0} %s {curve x%s 0} 
                            1 0 -15 -15 15 15 -10 -10 10 10 {} {}  {}  {}  {}  {}  {}  {}  {}  {}  {}   }''' % (    frame_start, name,
                                                                                                                    frame_start, x_str,
                                                                                                                    frame_start, y_str,
                                                                                                                    frame_start,
                                                                                                                    frame_start,
                                                                                                                    trs_val,
                                                                                                                    frame_start
                                                                                                                    )
        track_data_lines.append(track_line)


    
    # Generate complete .nk file content
    nk_content = f'''Tracker4 {{
tracks {{ {{ 1 31 {num_tracks} }} 
{{ {{ 5 1 20 enable e 1 }} 
{{ 3 1 75 name name 1 }} 
{{ 2 1 58 track_x track_x 1 }} 
{{ 2 1 58 track_y track_y 1 }} 
{{ 2 1 63 offset_x offset_x 1 }} 
{{ 2 1 63 offset_y offset_y 1 }} 
{{ 4 1 27 T T 1 }} 
{{ 4 1 27 R R 1 }} 
{{ 4 1 27 S S 1 }} 
{{ 2 0 45 error error 1 }} 
{{ 1 1 0 error_min error_min 1 }} 
{{ 1 1 0 error_max error_max 1 }} 
{{ 1 1 0 pattern_x pattern_x 1 }} 
{{ 1 1 0 pattern_y pattern_y 1 }} 
{{ 1 1 0 pattern_r pattern_r 1 }} 
{{ 1 1 0 pattern_t pattern_t 1 }} 
{{ 1 1 0 search_x search_x 1 }} 
{{ 1 1 0 search_y search_y 1 }} 
{{ 1 1 0 search_r search_r 1 }} 
{{ 1 1 0 search_t search_t 1 }} 
{{ 2 1 0 key_track key_track 1 }} 
{{ 2 1 0 key_search_x key_search_x 1 }} 
{{ 2 1 0 key_search_y key_search_y 1 }} 
{{ 2 1 0 key_search_r key_search_r 1 }} 
{{ 2 1 0 key_search_t key_search_t 1 }} 
{{ 2 1 0 key_track_x key_track_x 1 }} 
{{ 2 1 0 key_track_y key_track_y 1 }} 
{{ 2 1 0 key_track_r key_track_r 1 }} 
{{ 2 1 0 key_track_t key_track_t 1 }} 
{{ 2 1 0 key_centre_offset_x key_centre_offset_x 1 }} 
{{ 2 1 0 key_centre_offset_y key_centre_offset_y 1 }} 
}} 
{{
{chr(10).join(track_data_lines)}
}}
}}
reference_frame {reference_frame}
name CoTracker_Generated_{num_tracks}pts
}}

'''
    
    # Write .nk file
    with open(output_path, 'w') as f:
        f.write(nk_content)
    
    print(f"SUCCESS: Generated {output_path}")
    print(f"   {num_tracks} tracks with {total_rows - filtered_rows} total keyframes")
    print(f"   Ready to load in Nuke!")
    print()
    
    return output_path


if __name__ == "__main__":
    import sys
    # CLI usage:
    # python generate_exact_nuke_file.py csv_path output_path frame_offset reference_frame_video image_height
    if len(sys.argv) >= 6:
        csv_path = sys.argv[1]
        output_path = sys.argv[2]
        frame_offset = int(sys.argv[3])
        reference_frame_video = int(sys.argv[4])
        image_height = int(sys.argv[5])

        reference_frame = reference_frame_video + frame_offset

        out = generate_nuke_file(
            csv_path=csv_path,
            output_path=output_path,
            image_height=image_height,
            min_confidence=0.5,
            frame_offset=frame_offset,
            reference_frame=reference_frame
        )
        print(os.path.abspath(out))
    else:
        print("Usage: python generate_exact_nuke_file.py <csv_path> <output_path> <frame_offset> <reference_frame_video> <image_height>")
        print("Example: python generate_exact_nuke_file.py coords.csv out.nk 0 1001 1080")
