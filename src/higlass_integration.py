import json
import uuid
import jsonschema
import subprocess
import logging
import re
import time
import os
import hashlib

# Basic HiGlass Viewconf Schema (can be expanded)
HIGLASS_SCHEMA = {
    "type": "object",
    "properties": {
        "editable": {"type": "boolean"},
        "viewEditable": {"type": "boolean"},
        "tracksEditable": {"type": "boolean"},
        "exportViewUrl": {"type": "string"},
        "zoomFixed": {"type": "boolean"},
        "trackSourceServers": {"type": "array", "items": {"type": "string"}},
        "views": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string"},
                    "initialXDomain": {"type": "array", "items": {"type": "number"}},
                    "initialYDomain": {"type": "array", "items": {"type": "number"}},
                    "tracks": {
                        "type": "object",
                        "properties": {
                            "top": {"type": "array", "items": {"type": "object"}},
                            "left": {"type": "array", "items": {"type": "object"}},
                            "center": {"type": "array", "items": {"type": "object"}},
                            "right": {"type": "array", "items": {"type": "object"}},
                            "bottom": {"type": "array", "items": {"type": "object"}}
                        },
                        "required": ["top", "left", "center", "right", "bottom"]
                    },
                    "layout": {"type": "object"}
                    # Add more view properties as needed
                },
                "required": ["uid", "initialXDomain", "initialYDomain", "tracks", "layout"]
            }
        },
        "zoomLocks": {"type": "object"},
        "locationLocks": {"type": "object"},
        "valueScaleLocks": {"type": "object"}
    },
    "required": ["editable", "exportViewUrl", "trackSourceServers", "views"]
}

# Define cache directory relative to the script or project root
# Adjust this path as needed
CACHE_DIR = ".higlass_cache"

# --- Caching Functions ---

def _generate_cache_key(params_tuple):
    """
    Generates a cache key based on a tuple of parameters.
    Uses SHA256 hash for a consistent and safe filename.

    Args:
        params_tuple (tuple): A tuple containing all relevant parameters
                              that define the view configuration uniquely.
                              Order matters.

    Returns:
        str: A unique cache key (SHA256 hash).
    """
    # Convert tuple to a stable string representation
    # Using json.dumps ensures consistent ordering for dicts within tuple
    # and handles different data types reasonably well.
    params_string = json.dumps(params_tuple, sort_keys=True)
    return hashlib.sha256(params_string.encode('utf-8')).hexdigest()

def save_to_cache(key, viewconf_json):
    """
    Saves the generated viewconf JSON string to the cache.

    Args:
        key (str): The cache key.
        viewconf_json (str): The JSON string to save.
    """
    if not os.path.exists(CACHE_DIR):
        try:
            os.makedirs(CACHE_DIR)
            logging.info(f"Created cache directory: {CACHE_DIR}")
        except OSError as e:
            logging.error(f"Failed to create cache directory {CACHE_DIR}: {e}")
            return # Don't proceed if cache dir creation fails

    cache_file_path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(cache_file_path, 'w') as f:
            f.write(viewconf_json)
        logging.info(f"Saved viewconf to cache: {cache_file_path}")
    except IOError as e:
        logging.error(f"Failed to write to cache file {cache_file_path}: {e}")

def load_from_cache(key):
    """
    Attempts to load a viewconf JSON string from the cache.

    Args:
        key (str): The cache key.

    Returns:
        str or None: The cached JSON string if found, otherwise None.
    """
    cache_file_path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'r') as f:
                logging.info(f"Loading viewconf from cache: {cache_file_path}")
                return f.read()
        except IOError as e:
            logging.error(f"Failed to read from cache file {cache_file_path}: {e}")
            return None
    else:
        logging.info(f"Cache miss for key: {key}")
        return None

# --- Main Workflow Function with Caching ---

def get_or_create_higlass_json(track_configs, view_params=None, base_params=None):
    """
    Gets a HiGlass JSON configuration from cache or generates, validates, and caches it.

    Args:
        track_configs (list): A list of dictionaries, where each dictionary contains
                              the parameters needed for add_loop_track or
                              add_intensity_track (e.g., 'type': 'loop', 
                              'tileset_uid': ..., 'server': ..., 'sample_id': ..., 'condition': ..., 
                              'options': {} (optional), ...).
                              Must include 'type' ('loop' or 'intensity').
        view_params (dict, optional): Parameters for the initial view (e.g., initial domains).
                                     Defaults to None.
        base_params (dict, optional): Parameters for create_base_viewconf.
                                     Defaults to None.

    Returns:
        str: The HiGlass view configuration as a JSON string.

    Raises:
        ValueError: If track_configs is invalid or missing required keys.
        jsonschema.ValidationError: If the generated viewconf is invalid.
    """

    # Define cache key based on input parameters
    # Important: Use a stable representation (e.g., sorted tuples/dicts)
    # We use a tuple of track_configs (converted to tuples of sorted items),
    # sorted view_params items, and sorted base_params items.
    def stable_dict_tuple(d):
        return tuple(sorted(d.items())) if d else tuple()

    stable_track_configs = tuple(stable_dict_tuple(tc) for tc in sorted(track_configs, key=lambda x: (x.get('tileset_uid', ''), x.get('type', ''))))
    stable_view_params = stable_dict_tuple(view_params)
    stable_base_params = stable_dict_tuple(base_params)

    params_tuple = (stable_track_configs, stable_view_params, stable_base_params)
    cache_key = _generate_cache_key(params_tuple)

    # 1. Try loading from cache
    cached_json = load_from_cache(cache_key)
    if cached_json:
        # Optional: Could add validation here to ensure cached JSON is still valid
        # try:
        #     validate_viewconf(json.loads(cached_json))
        # except Exception as e:
        #     logging.warning(f"Cached viewconf for key {cache_key} failed validation: {e}. Regenerating.")
        # else:
        #     return cached_json
        return cached_json

    # 2. If not cached, generate the viewconf
    logging.info(f"Generating new viewconf for key: {cache_key}")
    if base_params is None: base_params = {}
    if view_params is None: view_params = {}

    vc = create_base_viewconf(**base_params)
    view_uid = add_view(vc, **view_params)

    for config in track_configs:
        track_type = config.get('type')
        common_args = {
            'viewconf': vc,
            'view_uid': view_uid,
            'tileset_uid': config.get('tileset_uid'),
            'server': config.get('server'),
            'sample_id': config.get('sample_id'),
            'condition': config.get('condition'),
            'options': config.get('options') # Pass options dict
        }
        # Filter out None values from common_args before passing
        common_args = {k: v for k, v in common_args.items() if v is not None}

        if not all(k in common_args for k in ['tileset_uid', 'server', 'sample_id', 'condition']):
             raise ValueError(f"Track config missing required keys (tileset_uid, server, sample_id, condition): {config}")

        try:
            if track_type == 'loop':
                add_loop_track(
                    height=config.get('height', 600),
                    width=config.get('width', 600),
                     **common_args
                )
            elif track_type == 'intensity':
                add_intensity_track(
                    track_suffix=config.get('track_suffix', "Intensity"),
                    height=config.get('height', 60),
                    **common_args
                )
            else:
                raise ValueError(f"Unsupported track type in config: {track_type}")
        except Exception as e:
             logging.error(f"Error adding track for config {config}: {e}")
             raise # Re-raise after logging

    # 3. Generate JSON (includes validation)
    try:
        generated_json = generate_json(vc)
    except jsonschema.ValidationError as e:
        logging.error(f"Generated viewconf failed validation: {e}")
        raise
    except Exception as e:
        logging.error(f"Error during JSON generation: {e}")
        raise

    # 4. Save to cache
    save_to_cache(cache_key, generated_json)

    return generated_json

def validate_viewconf(viewconf, schema=HIGLASS_SCHEMA):
    """
    Validates the HiGlass view configuration against the schema.

    Args:
        viewconf (dict): The view configuration dictionary.
        schema (dict): The JSON schema to validate against.

    Raises:
        jsonschema.ValidationError: If the viewconf is invalid.
    """
    try:
        jsonschema.validate(instance=viewconf, schema=schema)
        print("Viewconf validation successful.")
    except jsonschema.ValidationError as e:
        print(f"Viewconf validation failed: {e}")
        raise # Re-raise the exception after printing

def create_base_viewconf(editable=True, export_url="/api/v1/viewconfs/", track_source_servers=None):
    """
    Creates the basic structure of a HiGlass view configuration.

    Args:
        editable (bool): Whether the viewconf is editable in the UI.
        export_url (str): The API endpoint for exporting/saving the viewconf.
        track_source_servers (list, optional): List of servers to fetch track data from.
                                             Defaults to ['http://higlass.io/api/v1'].

    Returns:
        dict: A dictionary representing the base HiGlass view configuration.
    """
    if track_source_servers is None:
        track_source_servers = ['http://higlass.io/api/v1']

    return {
        "editable": editable,
        "viewEditable": True,
        "tracksEditable": True,
        "exportViewUrl": export_url,
        "zoomFixed": False,
        "trackSourceServers": track_source_servers,
        "views": [],
        "zoomLocks": {"locksByViewUid": {}, "locksDict": {}},
        "locationLocks": {"locksByViewUid": {}, "locksDict": {}},
        "valueScaleLocks": {"locksByViewUid": {}, "locksDict": {}}
    }

def add_view(viewconf, initial_x_domain=None, initial_y_domain=None, layout=None, uid=None):
    """
    Adds a new view to the HiGlass view configuration.

    Args:
        viewconf (dict): The existing HiGlass view configuration dictionary.
        initial_x_domain (list, optional): Initial X domain [start, end]. Defaults to [0, 3.1e9].
        initial_y_domain (list, optional): Initial Y domain [start, end]. If None, uses initial_x_domain.
        layout (dict, optional): Layout object (w, h, x, y, i). Defaults to a standard layout.
        uid (str, optional): Unique identifier for the view. Defaults to a generated UUID.

    Returns:
        str: The UID of the newly added view.
    """
    if uid is None:
        uid = str(uuid.uuid4())

    if initial_x_domain is None:
        # Default roughly covers hg19/hg38
        initial_x_domain = [0, 3100000000]
    if initial_y_domain is None:
        initial_y_domain = initial_x_domain

    if layout is None:
        # Basic default layout
        layout = {'w': 12, 'h': 12, 'x': 0, 'y': 0, 'i': uid, 'moved': False, 'static': False}
    else:
        layout['i'] = uid # Ensure layout 'i' matches view uid

    new_view = {
        "uid": uid,
        "initialXDomain": initial_x_domain,
        "initialYDomain": initial_y_domain,
        "tracks": {
            "top": [],
            "left": [],
            "center": [],
            "right": [],
            "bottom": []
        },
        "layout": layout,
        # Add default genome position search box if needed?
        # "genomePositionSearchBox": { ... }
    }

    viewconf['views'].append(new_view)
    return uid

def add_track(viewconf, view_uid, position, track_type, tileset_uid=None, server=None, data_config=None, options=None, height=100, width=100, uid=None):
    """
    Adds a track to a specific view and position within the view configuration.

    Args:
        viewconf (dict): The existing HiGlass view configuration dictionary.
        view_uid (str): The UID of the view to add the track to.
        position (str): The position ('top', 'left', 'center', 'right', 'bottom').
        track_type (str): The type of the track (e.g., 'heatmap', 'line', 'bedlike').
        tileset_uid (str, optional): The UID of the dataset on the HiGlass server. Required if data_config is not provided.
        server (str, optional): The URL of the HiGlass server. Required if data_config is not provided.
        data_config (dict, optional): Alternative data configuration (e.g., for local files, genbank).
        options (dict, optional): Track-specific options (e.g., color, valueScaleMin/Max). Defaults to {}.
        height (int): The height of the track.
        width (int): The width of the track.
        uid (str, optional): Unique identifier for the track. Defaults to a generated UUID.

    Returns:
        str: The UID of the newly added track.

    Raises:
        ValueError: If the specified view_uid or position is invalid, or if data source info is missing.
    """
    if uid is None:
        uid = str(uuid.uuid4())

    if options is None:
        options = {}

    track_definition = {
        "uid": uid,
        "type": track_type,
        "options": options,
        "height": height,
        "width": width
    }

    if data_config:
        track_definition["data"] = data_config
    elif tileset_uid and server:
        track_definition["tilesetUid"] = tileset_uid
        track_definition["server"] = server
    else:
        raise ValueError("Either data_config or both tileset_uid and server must be provided.")

    view_found = False
    for view in viewconf['views']:
        if view['uid'] == view_uid:
            view_found = True
            if position in view['tracks']:
                view['tracks'][position].append(track_definition)
                return uid
            else:
                raise ValueError(f"Invalid track position: {position}")

    if not view_found:
        raise ValueError(f"View with UID {view_uid} not found in viewconf.")

def add_loop_track(viewconf, view_uid, tileset_uid, server, sample_id, condition, options=None, height=600, width=600, uid=None):
    """
    Adds a 2D rectangle domain track for visualizing loops with sample/condition label.

    Args:
        viewconf (dict): The HiGlass view configuration.
        view_uid (str): UID of the view to add the track to.
        tileset_uid (str): Tileset UID for the loop data (bed2ddb format).
        server (str): HiGlass server URL.
        sample_id (str): Identifier for the sample.
        condition (str): Experimental condition.
        options (dict, optional): Additional track-specific options. Defaults will be used if None.
        height (int): Track height.
        width (int): Track width.
        uid (str, optional): Track UID.

    Returns:
        str: The UID of the added loop track.
    """
    track_name = f"{sample_id}_{condition}_Loops"
    default_options = {
        "labelPosition": "hidden", # Often hidden for center tracks
        "labelColor": "black",
        "labelTextOpacity": 0.4,
        "name": track_name,
        "colorRange": [ # Example color range, can be customized
            "#FF0000", # Red for lower values (e.g., less significant loops)
            "#0000FF"  # Blue for higher values (e.g., more significant loops)
        ],
        "minHeight": 10, # Minimum height for visibility
        "maxZoom": None,
        "flipDiagonal": "no" # 'yes', 'no', or 'copy'
        # Add other relevant 2d-rectangle-domains options as needed
    }
    if options:
        # Allow overriding the generated name if explicitly provided in options
        if 'name' not in options:
            options['name'] = track_name
        default_options.update(options)
    else:
        # Ensure the name is set if options is None
        options = default_options

    return add_track(
        viewconf=viewconf,
        view_uid=view_uid,
        position="center",
        track_type="2d-rectangle-domains",
        tileset_uid=tileset_uid,
        server=server,
        options=options, # Use the potentially updated options dictionary
        height=height,
        width=width,
        uid=uid
    )

def add_intensity_track(viewconf, view_uid, tileset_uid, server, sample_id, condition, track_suffix="Intensity", options=None, height=60, uid=None):
    """
    Adds a line track for visualizing 1D intensity data with sample/condition label.

    Args:
        viewconf (dict): The HiGlass view configuration.
        view_uid (str): UID of the view to add the track to.
        tileset_uid (str): Tileset UID for the intensity data (e.g., bigwig).
        server (str): HiGlass server URL.
        sample_id (str): Identifier for the sample.
        condition (str): Experimental condition.
        track_suffix (str): Suffix for the track name (default: "Intensity").
        options (dict, optional): Additional track-specific options. Defaults will be used if None.
        height (int): Track height.
        uid (str, optional): Track UID.

    Returns:
        str: The UID of the added intensity track.
    """
    track_name = f"{sample_id}_{condition}_{track_suffix}"
    default_options = {
        "labelPosition": "topLeft",
        "labelColor": "black",
        "labelTextOpacity": 0.4,
        "name": track_name,
        "axisPositionHorizontal": "right",
        "lineStrokeWidth": 1,
        "lineStrokeColor": "blue", # Default color, can be customized
        "valueScaling": "linear", # Or 'log'
        "minHeight": 20 # Minimum height for visibility
        # Add other relevant line track options as needed
    }
    if options:
        # Allow overriding the generated name if explicitly provided in options
        if 'name' not in options:
            options['name'] = track_name
        default_options.update(options)
    else:
        # Ensure the name is set if options is None
        options = default_options

    # Line tracks typically go in 'top' or 'bottom'
    return add_track(
        viewconf=viewconf,
        view_uid=view_uid,
        position="top", # Default to top, could be made a parameter
        track_type="line",
        tileset_uid=tileset_uid,
        server=server,
        options=options, # Use the potentially updated options dictionary
        height=height,
        # Width is usually determined by the view for top/bottom tracks
        uid=uid
    )

def generate_json(viewconf):
    """
    Generates the final JSON string from the view configuration dictionary,
    after validating it against the schema.

    Args:
        viewconf (dict): The completed HiGlass view configuration dictionary.

    Returns:
        str: A JSON string representation of the view configuration.

    Raises:
        jsonschema.ValidationError: If the viewconf is invalid.
    """
    validate_viewconf(viewconf) # Validate before generating JSON
    return json.dumps(viewconf, indent=2)

def ingest_tileset(file_path, file_type, datatype, assembly=None, name=None, uid=None, higlass_manage_path="higlass-manage", max_retries=3, initial_delay=5):
    """
    Ingests a dataset into HiGlass using the higlass-manage command-line tool,
    with exponential backoff logic for handling `subprocess.CalledProcessError` and logging retry attempts.

    Args:
        file_path (str): Path to the data file to ingest.
        file_type (str): Type of the file (e.g., 'cooler', 'bigwig', 'bed2ddb').
        datatype (str): HiGlass datatype (e.g., 'matrix', 'vector', '2d-rectangle-domains').
        assembly (str, optional): Genome assembly (e.g., 'hg19', 'hg38'). Defaults to None.
        name (str, optional): Name for the dataset in HiGlass. Defaults to the filename.
        uid (str, optional): Specify a UID for the dataset. Defaults to None (HiGlass generates one).
        higlass_manage_path (str): Path to the higlass-manage executable.
        max_retries (int): Maximum number of retry attempts for the command.
        initial_delay (int): Initial delay in seconds for exponential backoff.

    Returns:
        str: The UID of the ingested tileset.

    Raises:
        RuntimeError: If the higlass-manage command fails or UID cannot be parsed.
        FileNotFoundError: If higlass-manage executable is not found.
    """
    command = [
        higlass_manage_path,
        "ingest",
        "--filetype", file_type,
        "--datatype", datatype,
        "--filepath", file_path,
    ]
    if assembly:
        command.extend(["--assembly", assembly])
    if name:
        command.extend(["--name", name])
    if uid:
        command.extend(["--uid", uid])

    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        logging.info(f"Running higlass-manage command (Attempt {attempt + 1}/{max_retries + 1}): {' '.join(command)}")
        try:
            # Using subprocess.run to execute the command
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            stdout = result.stdout
            stderr = result.stderr
            logging.info(f"higlass-manage stdout:\n{stdout}")
            if stderr:
                logging.warning(f"higlass-manage stderr:\n{stderr}")

            # Attempt to parse the UID from stdout
            # Example output: "Tileset ingested: OHJakQICQD6gTD7skx4EWA"
            # Or more complex JSON output in some versions
            match = re.search(r"Tileset ingested:?\s*([\w\-]+)", stdout)
            if match:
                tileset_uid = match.group(1)
                logging.info(f"Successfully ingested tileset. UID: {tileset_uid}")
                return tileset_uid
            else:
                try:
                    output_data = json.loads(stdout)
                    if isinstance(output_data, dict) and 'uuid' in output_data:
                        tileset_uid = output_data['uuid']
                        logging.info(f"Successfully ingested tileset (parsed from JSON). UID: {tileset_uid}")
                        return tileset_uid
                except json.JSONDecodeError:
                    pass

                # If parsing fails even after successful command run
                logging.error(f"Could not parse tileset UID from higlass-manage output (Attempt {attempt + 1}): {stdout}")
                # Decide whether to retry parsing failure or raise immediately
                # For now, let's raise immediately if parsing fails after success
                raise RuntimeError("Failed to parse tileset UID from higlass-manage output even after successful command execution.")

        except FileNotFoundError as e:
            logging.error(f"higlass-manage executable not found at '{higlass_manage_path}'. Please ensure it's installed and in the system PATH or provide the correct path.")
            raise # Do not retry if executable is not found
        except subprocess.CalledProcessError as e:
            logging.warning(f"higlass-manage command failed (Attempt {attempt + 1}) with exit code {e.returncode}.")
            logging.warning(f"Stderr:\n{e.stderr}")
            logging.warning(f"Stdout:\n{e.stdout}")
            last_exception = e
            if attempt < max_retries:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2 # Exponential backoff
            else:
                logging.error("Maximum retries reached. Command failed.")
                raise RuntimeError(f"higlass-manage command failed after {max_retries + 1} attempts: {e.stderr}") from e
        except Exception as e:
            # Catch other unexpected errors during the process
            logging.error(f"An unexpected error occurred during ingestion (Attempt {attempt + 1}): {e}")
            last_exception = e
            if attempt < max_retries:
                 logging.info(f"Retrying in {delay} seconds due to unexpected error...")
                 time.sleep(delay)
                 delay *= 2
            else:
                logging.error("Maximum retries reached after unexpected error.")
                raise # Re-raise the last unexpected error

    # This part should theoretically not be reached if logic is correct,
    # but added as a safeguard.
    logging.error("Ingestion failed after all retries.")
    if last_exception:
        raise RuntimeError("Ingestion failed after all retries") from last_exception
    else:
        raise RuntimeError("Ingestion failed after all retries for an unknown reason.")

# --- Example Usage (can be removed later) ---
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define track configurations (using previously defined example UIDs/params)
    example_loop_uid = "dummy-loop-tileset"
    example_bw_uid = "AS_6-Me12lSAhLo1QB31Mg"
    example_server = "http://higlass.io/api/v1"
    example_sample = "SampleA"
    example_condition = "Control"

    track_set1 = [
        {
            'type': 'loop',
            'tileset_uid': example_loop_uid,
            'server': example_server,
            'sample_id': example_sample,
            'condition': example_condition,
            'options': {"colorRange": ["#CCCCCC", "#0000FF"]}
        },
        {
            'type': 'intensity',
            'tileset_uid': example_bw_uid,
            'server': example_server,
            'sample_id': example_sample,
            'condition': example_condition,
            'options': {"lineStrokeColor": "green"}
        }
    ]

    # Example of getting/creating the viewconf
    print("\n--- Generating/Caching ViewConf (Attempt 1) ---")
    try:
        json_output1 = get_or_create_higlass_json(track_set1)
        print("Generated/Cached JSON Output (1):")
        # print(json_output1) # Optionally print full JSON
        data1 = json.loads(json_output1)
        print(f"View 1 UID: {data1['views'][0]['uid']}")
        print(f"Num Top Tracks: {len(data1['views'][0]['tracks']['top'])}")
        print(f"Num Center Tracks: {len(data1['views'][0]['tracks']['center'])}")

    except Exception as e:
        print(f"Error getting/creating viewconf: {e}")

    print("\n--- Generating/Caching ViewConf (Attempt 2 - should load from cache) ---")
    try:
        # Use the exact same parameters to hit the cache
        json_output2 = get_or_create_higlass_json(track_set1)
        print("Generated/Cached JSON Output (2):")
        # print(json_output2)
        data2 = json.loads(json_output2)
        print(f"View 2 UID: {data2['views'][0]['uid']}") # Should be same as View 1
        print(f"Num Top Tracks: {len(data2['views'][0]['tracks']['top'])}")
        print(f"Num Center Tracks: {len(data2['views'][0]['tracks']['center'])}")

        # Verify content is the same (ignoring potential minor whitespace diffs if any)
        assert json.loads(json_output1) == json.loads(json_output2), "Cached and regenerated JSON differ!"
        print("Cache hit verified successfully.")

    except Exception as e:
        print(f"Error getting/creating viewconf (Attempt 2): {e}")

    # Example with different parameters (should generate a new cache entry)
    print("\n--- Generating/Caching ViewConf (Different Params) ---")
    track_set2 = [
        {
            'type': 'intensity',
            'tileset_uid': 'ANOTHER_BW_UID',
            'server': example_server,
            'sample_id': 'SampleB',
            'condition': 'Treated'
        }
    ]
    try:
        json_output3 = get_or_create_higlass_json(track_set2)
        print("Generated/Cached JSON Output (3):")
        data3 = json.loads(json_output3)
        print(f"View 3 UID: {data3['views'][0]['uid']}")
        print(f"Num Top Tracks: {len(data3['views'][0]['tracks']['top'])}")
        print(f"Num Center Tracks: {len(data3['views'][0]['tracks']['center'])}")
    except Exception as e:
        print(f"Error getting/creating viewconf (Different Params): {e}")

    # --- Ingestion Example (requires setup) ---
    # ... (ingestion example remains the same, might move it inside the main workflow eventually)

    # --- Original Example (now superseded by get_or_create_higlass_json) ---
    # ... (Can be removed or kept for reference) ... 