"""
write python program to analyze and report on all photos under the directory O:\Bilder\1-d7100. Store the results in a Duckdb database called photos_db.db and include statistics on what kind of photos, e.g., landscape, birds, people, water, forest, spring, summer, autumn, winter along with camerra settings, date and time taken. Provide graphs showing type of photo, year taken, camera settings
"""
#!/usr/bin/env python3
"""
Photo analysis and reporting pipeline.
Scans directory, extracts EXIF, performs image tagging, stores results in DuckDB,
and creates summary graphs.
Author: Copilot
"""
