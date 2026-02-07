import os
import timescape_functions as tf

scratch = os.environ.get("_CONDOR_SCRATCH_DIR", ".")

zoo_dir = f"{scratch}/ZOO"
sdss_dir = f"{scratch}/SDSS"

#os.makedirs(zoo_dir, exist_ok=True)
os.makedirs(sdss_dir, exist_ok=True)

try:
    tf.loop_galaxy_chunk(
        tf.galaxy_zoo_chunk_query, 500, 0, "galaxy_zoo_morphology_until_", "full_morphology.csv", zoo_dir,
    )
except Exception as e:
    print(f"ZOO loop failed: {e}")

try:
    tf.loop_galaxy_chunk(
        tf.sdss_chunk_query, 500, 0, "sdss_db_", "full_sdss.csv", sdss_dir,
    )
except Exception as e:
    print(f"SDSS loop failed: {e}")