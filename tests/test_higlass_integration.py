import unittest
import json
import uuid
import jsonschema
import subprocess
from unittest.mock import patch, MagicMock, call, mock_open
import time
import os
import hashlib

# Assuming higlass_integration.py is in the src directory and tests is adjacent
# Adjust the import path as necessary based on your project structure
from src.higlass_integration import (
    create_base_viewconf,
    add_view,
    add_track,
    add_loop_track,
    add_intensity_track,
    generate_json,
    validate_viewconf,
    HIGLASS_SCHEMA,
    ingest_tileset,
    _generate_cache_key,
    save_to_cache,
    load_from_cache,
    get_or_create_higlass_json,
    CACHE_DIR
)

# --- Helper data for cache tests ---
DUMMY_TRACK_CONFIG_1 = {
    'type': 'loop',
    'tileset_uid': 'loops1',
    'server': 'server1',
    'sample_id': 'S1',
    'condition': 'C1',
    'options': {"a": 1}
}
DUMMY_TRACK_CONFIG_2 = {
    'type': 'intensity',
    'tileset_uid': 'intensity1',
    'server': 'server1',
    'sample_id': 'S1',
    'condition': 'C1',
    'options': {"b": 2}
}
DUMMY_PARAMS_TUPLE = (
    (
        (('condition', 'C1'), ('options', (('a', 1),)), ('sample_id', 'S1'), ('server', 'server1'), ('tileset_uid', 'loops1'), ('type', 'loop')),
        (('condition', 'C1'), ('options', (('b', 2),)), ('sample_id', 'S1'), ('server', 'server1'), ('tileset_uid', 'intensity1'), ('type', 'intensity'))
    ),
    tuple(),
    tuple()
)
DUMMY_JSON_OUTPUT = json.dumps({"editable": True, "views": []})

class TestHiglassIntegration(unittest.TestCase):

    def test_create_base_viewconf(self):
        vc = create_base_viewconf()
        self.assertTrue(vc['editable'])
        self.assertIn('http://higlass.io/api/v1', vc['trackSourceServers'])
        self.assertEqual(vc['views'], [])
        self.assertIn('exportViewUrl', vc)
        # Check if it conforms to the basic schema structure
        validate_viewconf(vc)

    def test_add_view(self):
        vc = create_base_viewconf()
        view_uid = add_view(vc)
        self.assertEqual(len(vc['views']), 1)
        self.assertEqual(vc['views'][0]['uid'], view_uid)
        self.assertIsNotNone(uuid.UUID(view_uid)) # Check if it's a valid UUID
        self.assertIn('initialXDomain', vc['views'][0])
        self.assertIn('initialYDomain', vc['views'][0])
        self.assertIn('layout', vc['views'][0])
        self.assertEqual(vc['views'][0]['layout']['i'], view_uid)
        validate_viewconf(vc)

    def test_add_track_valid(self):
        vc = create_base_viewconf()
        view_uid = add_view(vc)
        track_uid = add_track(
            vc,
            view_uid,
            position="top",
            track_type="line",
            tileset_uid="dummy_tileset",
            server="dummy_server",
            options={"name": "Test Track"},
            height=50
        )
        self.assertEqual(len(vc['views'][0]['tracks']['top']), 1)
        track = vc['views'][0]['tracks']['top'][0]
        self.assertEqual(track['uid'], track_uid)
        self.assertEqual(track['type'], "line")
        self.assertEqual(track['tilesetUid'], "dummy_tileset")
        self.assertEqual(track['server'], "dummy_server")
        self.assertEqual(track['options']['name'], "Test Track")
        self.assertEqual(track['height'], 50)
        validate_viewconf(vc)

    def test_add_track_data_config(self):
        vc = create_base_viewconf()
        view_uid = add_view(vc)
        data_config = {"type": "vector", "url": "http://example.com/data.bw", "filetype": "bigwig"}
        track_uid = add_track(
            vc,
            view_uid,
            position="top",
            track_type="line",
            data_config=data_config,
            options={"name": "Test Data Track"}
        )
        track = vc['views'][0]['tracks']['top'][0]
        self.assertEqual(track['uid'], track_uid)
        self.assertEqual(track['type'], "line")
        self.assertEqual(track['data'], data_config)
        self.assertNotIn('tilesetUid', track)
        self.assertNotIn('server', track)
        # Basic schema validation might need expansion to cover data config details
        # validate_viewconf(vc)

    def test_add_track_invalid_view(self):
        vc = create_base_viewconf()
        with self.assertRaisesRegex(ValueError, "View with UID invalid_view_uid not found"): # Exact message match
            add_track(vc, "invalid_view_uid", "top", "line", "dummy_tileset", "dummy_server")

    def test_add_track_invalid_position(self):
        vc = create_base_viewconf()
        view_uid = add_view(vc)
        with self.assertRaisesRegex(ValueError, "Invalid track position: invalid_pos"): # Exact message match
            add_track(vc, view_uid, "invalid_pos", "line", "dummy_tileset", "dummy_server")

    def test_add_track_missing_data_source(self):
        vc = create_base_viewconf()
        view_uid = add_view(vc)
        with self.assertRaisesRegex(ValueError, "Either data_config or both tileset_uid and server must be provided."): # Exact message match
            add_track(vc, view_uid, "top", "line")

    def test_add_loop_track(self):
        vc = create_base_viewconf()
        view_uid = add_view(vc)
        track_uid = add_loop_track(vc, view_uid, "loop_tileset", "loop_server", "SampleA", "Control", options={"custom": "value"})
        self.assertEqual(len(vc['views'][0]['tracks']['center']), 1)
        track = vc['views'][0]['tracks']['center'][0]
        self.assertEqual(track['type'], "2d-rectangle-domains")
        self.assertEqual(track['tilesetUid'], "loop_tileset")
        self.assertEqual(track['server'], "loop_server")
        self.assertEqual(track['options']['name'], "SampleA_Control_Loops") # Check generated name
        self.assertEqual(track['options']['custom'], "value") # Check merged options
        validate_viewconf(vc)

    def test_add_loop_track_override_name(self):
        vc = create_base_viewconf()
        view_uid = add_view(vc)
        track_uid = add_loop_track(vc, view_uid, "loop_tileset", "loop_server", "SampleB", "Treated", options={"name": "Explicit Name"})
        track = vc['views'][0]['tracks']['center'][0]
        self.assertEqual(track['options']['name'], "Explicit Name") # Check overridden name
        validate_viewconf(vc)

    def test_add_intensity_track(self):
        vc = create_base_viewconf()
        view_uid = add_view(vc)
        track_uid = add_intensity_track(vc, view_uid, "intensity_tileset", "intensity_server", "SampleX", "ConditionY", track_suffix="Signal", options={"color": "red"})
        self.assertEqual(len(vc['views'][0]['tracks']['top']), 1)
        track = vc['views'][0]['tracks']['top'][0]
        self.assertEqual(track['type'], "line")
        self.assertEqual(track['tilesetUid'], "intensity_tileset")
        self.assertEqual(track['server'], "intensity_server")
        self.assertEqual(track['options']['name'], "SampleX_ConditionY_Signal") # Check generated name
        self.assertEqual(track['options']['lineStrokeColor'], "blue") # Check default color
        self.assertEqual(track['options']['color'], "red") # Check merged options
        validate_viewconf(vc)

    def test_add_intensity_track_override_name(self):
        vc = create_base_viewconf()
        view_uid = add_view(vc)
        track_uid = add_intensity_track(vc, view_uid, "intensity_tileset", "intensity_server", "SampleZ", "Control", options={"name": "Specific Intensity"})
        track = vc['views'][0]['tracks']['top'][0]
        self.assertEqual(track['options']['name'], "Specific Intensity") # Check overridden name
        validate_viewconf(vc)

    def test_validate_viewconf_valid(self):
        vc = create_base_viewconf()
        view_uid = add_view(vc)
        add_track(vc, view_uid, "top", "line", "dummy_tileset", "dummy_server")
        try:
            validate_viewconf(vc)
        except jsonschema.ValidationError:
            self.fail("validate_viewconf raised ValidationError unexpectedly!")

    def test_validate_viewconf_invalid(self):
        invalid_vc = {"views": [{}] } # Missing required top-level fields
        with self.assertRaises(jsonschema.ValidationError):
            validate_viewconf(invalid_vc)

    def test_generate_json(self):
        vc = create_base_viewconf()
        view_uid = add_view(vc)
        add_track(vc, view_uid, "top", "line", "dummy_tileset", "dummy_server")
        json_output = generate_json(vc)
        # Check if it's valid JSON
        try:
            data = json.loads(json_output)
            self.assertIsInstance(data, dict)
        except json.JSONDecodeError:
            self.fail("generate_json did not produce valid JSON")
        # Check if the structure roughly matches
        self.assertIn("views", data)
        self.assertEqual(len(data['views']), 1)
        self.assertEqual(len(data['views'][0]['tracks']['top']), 1)

    def test_generate_json_invalid_conf(self):
        invalid_vc = {"views": [{}] } # Invalid structure
        with self.assertRaises(jsonschema.ValidationError):
            generate_json(invalid_vc) # Should raise validation error

    # --- Tests for Caching --- 

    def test_generate_cache_key_consistency(self):
        key1 = _generate_cache_key(DUMMY_PARAMS_TUPLE)
        key2 = _generate_cache_key(DUMMY_PARAMS_TUPLE) # Identical input
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 64) # SHA256 length

    def test_generate_cache_key_difference(self):
        key1 = _generate_cache_key(DUMMY_PARAMS_TUPLE)
        # Create slightly different params tuple
        modified_tuple = (
             DUMMY_PARAMS_TUPLE[0], # Same tracks
             (('initialXDomain', [0, 1000]),), # Different view params
             DUMMY_PARAMS_TUPLE[2]
        )
        key2 = _generate_cache_key(modified_tuple)
        self.assertNotEqual(key1, key2)

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_to_cache_new_dir(self, mock_file, mock_exists, mock_makedirs):
        """Test saving to cache when directory doesn't exist."""
        mock_exists.return_value = False # Simulate cache dir not existing
        key = "testkey123"
        data = '{"data": "test"}'
        expected_path = os.path.join(CACHE_DIR, f"{key}.json")

        save_to_cache(key, data)

        mock_exists.assert_called_once_with(CACHE_DIR)
        mock_makedirs.assert_called_once_with(CACHE_DIR)
        mock_file.assert_called_once_with(expected_path, 'w')
        mock_file().write.assert_called_once_with(data)

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_to_cache_existing_dir(self, mock_file, mock_exists, mock_makedirs):
        """Test saving to cache when directory already exists."""
        mock_exists.return_value = True # Simulate cache dir existing
        key = "testkey456"
        data = '{"other": "data"}'
        expected_path = os.path.join(CACHE_DIR, f"{key}.json")

        save_to_cache(key, data)

        mock_exists.assert_called_once_with(CACHE_DIR)
        mock_makedirs.assert_not_called() # Should not be called if exists
        mock_file.assert_called_once_with(expected_path, 'w')
        mock_file().write.assert_called_once_with(data)

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"cached": true}')
    def test_load_from_cache_hit(self, mock_file, mock_exists):
        """Test loading from cache when file exists."""
        mock_exists.return_value = True
        key = "cachehitkey"
        expected_path = os.path.join(CACHE_DIR, f"{key}.json")

        result = load_from_cache(key)

        mock_exists.assert_called_once_with(expected_path)
        mock_file.assert_called_once_with(expected_path, 'r')
        self.assertEqual(result, '{"cached": true}')

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_from_cache_miss(self, mock_file, mock_exists):
        """Test loading from cache when file does not exist."""
        mock_exists.return_value = False
        key = "cachemisskey"
        expected_path = os.path.join(CACHE_DIR, f"{key}.json")

        result = load_from_cache(key)

        mock_exists.assert_called_once_with(expected_path)
        mock_file.assert_not_called()
        self.assertIsNone(result)

    @patch('src.higlass_integration.load_from_cache')
    @patch('src.higlass_integration.save_to_cache')
    @patch('src.higlass_integration.generate_json') # Mock generation/validation
    def test_get_or_create_cache_miss(self, mock_generate, mock_save, mock_load):
        """Test the main workflow on a cache miss."""
        mock_load.return_value = None # Simulate cache miss
        mock_generate.return_value = DUMMY_JSON_OUTPUT # Simulate generation

        track_configs = [DUMMY_TRACK_CONFIG_1, DUMMY_TRACK_CONFIG_2]
        result = get_or_create_higlass_json(track_configs)

        expected_key = _generate_cache_key(DUMMY_PARAMS_TUPLE)

        mock_load.assert_called_once_with(expected_key)
        mock_generate.assert_called_once() # Check that generation was called
        mock_save.assert_called_once_with(expected_key, DUMMY_JSON_OUTPUT)
        self.assertEqual(result, DUMMY_JSON_OUTPUT)

    @patch('src.higlass_integration.load_from_cache')
    @patch('src.higlass_integration.save_to_cache')
    @patch('src.higlass_integration.generate_json') # Mock generation/validation
    def test_get_or_create_cache_hit(self, mock_generate, mock_save, mock_load):
        """Test the main workflow on a cache hit."""
        cached_data = '{"cached": true, "value": 123}'
        mock_load.return_value = cached_data # Simulate cache hit

        track_configs = [DUMMY_TRACK_CONFIG_1, DUMMY_TRACK_CONFIG_2]
        result = get_or_create_higlass_json(track_configs)

        expected_key = _generate_cache_key(DUMMY_PARAMS_TUPLE)

        mock_load.assert_called_once_with(expected_key)
        mock_generate.assert_not_called() # Should NOT be called on cache hit
        mock_save.assert_not_called()     # Should NOT be called on cache hit
        self.assertEqual(result, cached_data)

    # --- Tests for ingest_tileset --- 

    @patch('subprocess.run')
    def test_ingest_tileset_success_regex(self, mock_run):
        """Test successful ingestion with UID parsed via regex."""
        expected_uid = "ABCDEFG12345"
        # Configure the mock to simulate a successful run
        mock_process = MagicMock()
        mock_process.stdout = f"Blah blah\nTileset ingested: {expected_uid}\nMore output"
        mock_process.stderr = ""
        mock_run.return_value = mock_process

        uid = ingest_tileset("/path/to/file.cool", "cooler", "matrix")

        self.assertEqual(uid, expected_uid)
        mock_run.assert_called_once()
        # Check basic command structure
        args, kwargs = mock_run.call_args
        self.assertIn("higlass-manage", args[0])
        self.assertIn("ingest", args[0])
        self.assertIn("--filepath", args[0])
        self.assertIn("/path/to/file.cool", args[0])
        self.assertTrue(kwargs['check'])
        self.assertTrue(kwargs['capture_output'])

    @patch('subprocess.run')
    def test_ingest_tileset_success_json(self, mock_run):
        """Test successful ingestion with UID parsed via JSON fallback."""
        expected_uid = "XYZ-JSON-UID"
        # Configure the mock for JSON output
        mock_process = MagicMock()
        mock_process.stdout = json.dumps({"uuid": expected_uid, "other_data": "stuff"})
        mock_process.stderr = ""
        mock_process.check_returncode.return_value = None
        mock_run.return_value = mock_process

        uid = ingest_tileset("/path/to/file.bw", "bigwig", "vector")

        self.assertEqual(uid, expected_uid)
        mock_run.assert_called_once()

    @patch('time.sleep') # Mock time.sleep
    @patch('subprocess.run')
    def test_ingest_tileset_retry_success(self, mock_run, mock_sleep):
        """Test successful ingestion after retrying."""
        expected_uid = "RETRY_SUCCESS_UID"
        fail_process = subprocess.CalledProcessError(returncode=1, cmd=['cmd'], stderr="Temporary failure")
        success_process = MagicMock()
        success_process.stdout = f"Tileset ingested: {expected_uid}"
        success_process.stderr = ""

        # Simulate failure twice, then success
        mock_run.side_effect = [fail_process, fail_process, success_process]

        # Use max_retries=3 for the test
        uid = ingest_tileset("file.cool", "cooler", "matrix", max_retries=3, initial_delay=1)

        self.assertEqual(uid, expected_uid)
        self.assertEqual(mock_run.call_count, 3) # Initial call + 2 retries
        self.assertEqual(mock_sleep.call_count, 2) # Sleep called twice
        # Check that sleep durations were exponential (1, 2)
        mock_sleep.assert_has_calls([call(1), call(2)])

    @patch('time.sleep') # Mock time.sleep
    @patch('subprocess.run')
    def test_ingest_tileset_retry_exhausted(self, mock_run, mock_sleep):
        """Test ingestion failure after exhausting all retries."""
        fail_process = subprocess.CalledProcessError(returncode=1, cmd=['cmd'], stderr="Persistent failure")

        # Simulate failure on all attempts (initial + retries)
        mock_run.side_effect = fail_process

        with self.assertRaisesRegex(RuntimeError, "higlass-manage command failed after 4 attempts: Persistent failure"):
            ingest_tileset("file.cool", "cooler", "matrix", max_retries=3, initial_delay=1)

        self.assertEqual(mock_run.call_count, 4) # Initial call + 3 retries
        self.assertEqual(mock_sleep.call_count, 3) # Sleep called 3 times
        mock_sleep.assert_has_calls([call(1), call(2), call(4)])

    @patch('subprocess.run')
    def test_ingest_tileset_cmd_fail(self, mock_run):
        """Test ingestion failure due to command error."""
        # Configure mock to raise CalledProcessError
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=['higlass-manage', '...'], stderr="Command failed!"
        )

        with self.assertRaisesRegex(RuntimeError, "higlass-manage command failed: Command failed!"):
            ingest_tileset("file.bed2ddb", "bed2ddb", "2d-rectangle-domains")

        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_ingest_tileset_parse_fail(self, mock_run):
        """Test ingestion failure due to unparseable UID."""
        # Configure mock for successful run but unparseable output
        mock_process = MagicMock()
        mock_process.stdout = "Ingestion completed but no UID found here."
        mock_process.stderr = ""
        mock_process.check_returncode.return_value = None
        mock_run.return_value = mock_process

        with self.assertRaisesRegex(RuntimeError, "Failed to parse tileset UID"):
            ingest_tileset("file.cool", "cooler", "matrix")

        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_ingest_tileset_file_not_found(self, mock_run):
        """Test ingestion failure due to higlass-manage not found."""
        # Configure mock to raise FileNotFoundError
        mock_run.side_effect = FileNotFoundError("higlass-manage not found")

        with self.assertRaises(FileNotFoundError):
            ingest_tileset("file.cool", "cooler", "matrix")

        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_ingest_tileset_command_args(self, mock_run):
        """Test if optional arguments are correctly added to the command."""
        expected_uid = "COMMAND_TEST_UID"
        mock_process = MagicMock()
        mock_process.stdout = f"Tileset ingested: {expected_uid}"
        mock_process.stderr = ""
        mock_process.check_returncode.return_value = None
        mock_run.return_value = mock_process

        uid = ingest_tileset(
            "/data/my.cool",
            file_type="cooler",
            datatype="matrix",
            assembly="hg38",
            name="MyTestDataset",
            uid="SPECIFIC_UID",
            higlass_manage_path="/opt/bin/hg-manage"
        )

        self.assertEqual(uid, expected_uid)
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        command = args[0]

        self.assertEqual(command[0], "/opt/bin/hg-manage")
        self.assertIn("--assembly", command)
        self.assertIn("hg38", command)
        self.assertIn("--name", command)
        self.assertIn("MyTestDataset", command)
        self.assertIn("--uid", command)
        self.assertIn("SPECIFIC_UID", command)
        self.assertIn("--filepath", command)
        self.assertIn("/data/my.cool", command)


if __name__ == '__main__':
    unittest.main() 