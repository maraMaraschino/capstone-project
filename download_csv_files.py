import timescape_functions as tf

tf.loop_galaxy_chunk(tf.galaxy_zoo_chunk_query, 500, 0, "galaxy_zoo_morphology_until_", "full_morphology", "ZOO")
tf.loop_galaxy_chunk(tf.sdss_chunk_query, 10000, 0, "sdss_db_", "full_sdss.csv", "SDSS")