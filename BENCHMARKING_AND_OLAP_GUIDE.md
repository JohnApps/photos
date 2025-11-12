================================================================================
BENCHMARKING & OLAP ANALYSIS GUIDE
================================================================================

This document covers using the photo analyzer with DuckDB for OLAP queries,
benchmarking across platforms, and scale testing inspired by TPC-DS concepts.

================================================================================
PART 1: OLAP QUERIES ON PHOTO METADATA
================================================================================

The generated DuckDB database is perfect for OLAP (Online Analytical Processing)
queries on photo metadata across multiple dimensions:

DIMENSION TABLE CONCEPT:
────────────────────────
• datetime_original → Date/Time dimension (year, month, day)
• make, model → Camera dimension
• categories → Subject dimension
• season → Temporal dimension
• gps_lat, gps_lon → Geographic dimension
• image_width, image_height → Media quality dimension

BASIC OLAP QUERIES:
───────────────────

1. BY CAMERA (Equipment Analysis)
   ──────────────────────────────
   SELECT 
       make, 
       model, 
       COUNT(*) as photo_count,
       AVG(iso) as avg_iso,
       AVG(aperture) as avg_aperture,
       AVG(shutter_speed) as avg_shutter
   FROM photos
   WHERE make IS NOT NULL AND model IS NOT NULL
   GROUP BY make, model
   ORDER BY photo_count DESC
   LIMIT 20;

   ✓ Understand equipment usage patterns
   ✓ Identify most used camera/lens combinations
   ✓ Analyze exposure patterns by camera

2. BY TIME (Temporal Analysis)
   ──────────────────────────
   SELECT 
       year,
       month,
       COUNT(*) as photos,
       COUNT(DISTINCT date_trunc('day', 
           strptime(datetime_original, '%Y-%m-%d %H:%M:%S'))) as days_active,
       ROUND(AVG(image_width * image_height) / 1000000.0, 2) as avg_megapixels
   FROM photos
   WHERE datetime_original IS NOT NULL
   GROUP BY year, month
   ORDER BY year DESC, month DESC;

   ✓ See shooting intensity over time
   ✓ Seasonal patterns in photography activity
   ✓ Image quality trends

3. BY SUBJECT (Content Analysis)
   ──────────────────────────────
   SELECT 
       category,
       COUNT(*) as photo_count,
       ROUND(100.0 * COUNT(*) / 
           (SELECT COUNT(*) FROM photos), 2) as percent_of_total
   FROM (
       SELECT UNNEST(categories) as category FROM photos
   )
   GROUP BY category
   ORDER BY photo_count DESC;

   ✓ What subjects are most photographed
   ✓ Content distribution across collection
   ✓ Percentage breakdown

4. BY SEASON (Environmental Analysis)
   ─────────────────────────────────
   SELECT 
       season,
       COUNT(*) as photo_count,
       AVG(CAST(image_width * image_height AS FLOAT)) as avg_pixels,
       COUNT(DISTINCT CASE WHEN iso > 3200 THEN filename END) as high_iso_photos
   FROM photos
   WHERE season IS NOT NULL
   GROUP BY season
   ORDER BY photo_count DESC;

   ✓ Seasonal photography patterns
   ✓ Technical settings by season
   ✓ Low-light shooting frequency

5. GEOGRAPHIC HOTSPOTS (GPS Analysis)
   ──────────────────────────────────
   SELECT 
       ROUND(gps_lat, 1) as lat_bucket,
       ROUND(gps_lon, 1) as lon_bucket,
       COUNT(*) as photos,
       COUNT(DISTINCT 
           date_trunc('day', strptime(datetime_original, '%Y-%m-%d %H:%M:%S'))
       ) as days_visited
   FROM photos
   WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL
   GROUP BY lat_bucket, lon_bucket
   ORDER BY photos DESC
   LIMIT 20;

   ✓ Most photographed locations
   ✓ Revisit frequency
   ✓ Geographic clustering analysis


ADVANCED OLAP: CUBE & ROLL-UP OPERATIONS
──────────────────────────────────────────

DuckDB supports cube-like operations for multi-dimensional analysis:

6. Multi-Dimensional Cube (Seasons × Make)
   ───────────────────────────────────────
   SELECT 
       COALESCE(season, 'ALL_SEASONS') as season,
       COALESCE(make, 'ALL_MAKES') as make,
       COUNT(*) as photos
   FROM photos
   WHERE season IS NOT NULL AND make IS NOT NULL
   GROUP BY ROLLUP(season, make)
   ORDER BY season, make;

   ✓ See data at different aggregation levels
   ✓ Compare within-category vs overall
   ✓ Identify anomalies in combinations

7. Cross-Tabulation: Camera Make vs Season
   ───────────────────────────────────────
   SELECT 
       make,
       SUM(CASE WHEN season = 'spring' THEN 1 ELSE 0 END) as spring,
       SUM(CASE WHEN season = 'summer' THEN 1 ELSE 0 END) as summer,
       SUM(CASE WHEN season = 'autumn' THEN 1 ELSE 0 END) as autumn,
       SUM(CASE WHEN season = 'winter' THEN 1 ELSE 0 END) as winter,
       COUNT(*) as total
   FROM photos
   WHERE make IS NOT NULL AND season IS NOT NULL
   GROUP BY make
   ORDER BY total DESC
   LIMIT 10;

   ✓ Pivot table style analysis
   ✓ See seasonal variations by equipment


================================================================================
PART 2: SCALE FACTORS & TPC-DS INSPIRED BENCHMARKING
================================================================================

TPC-DS defines scale factors (1GB, 10GB, 100GB, etc.). We apply similar
concepts to photo metadata analysis:

SCALE FACTOR DEFINITION:
────────────────────────
• SF 0.1  = 100 files
• SF 1.0  = 1,000 files (default/production)
• SF 5.0  = 5,000 files (large batch)
• SF 10.0 = 10,000 files (archive scale)

BENCHMARK TEST SUITE:
─────────────────────

Run these commands to generate benchmark datasets:

# SF 0.1 - Quick test
python photo_analyze_jpg_only.py --root-dir ./photos --max-files 100 \
    --db-path bench_sf01.db

# SF 1.0 - Standard
python photo_analyze_jpg_only.py --root-dir ./photos --max-files 1000 \
    --db-path bench_sf10.db

# SF 5.0 - Large batch
python photo_analyze_jpg_only.py --root-dir ./photos --max-files 5000 \
    --db-path bench_sf50.db

# SF 10.0 - Archive scale
python photo_analyze_jpg_only.py --root-dir ./photos --max-files 10000 \
    --db-path bench_sf100.db


QUERY PERFORMANCE BENCHMARKING:
────────────────────────────────

Use this to measure query performance across scale factors:

BENCHMARK QUERY 1: Full Table Scan
──────────────────────────────────
duckdb bench_sf01.db -c ".timer on" \
  "SELECT COUNT(*), AVG(image_width), AVG(image_height) FROM photos;"

Expected performance:
  SF 0.1 (100):    < 10 ms
  SF 1.0 (1K):     10-50 ms
  SF 5.0 (5K):     50-200 ms
  SF 10.0 (10K):   100-500 ms

BENCHMARK QUERY 2: Grouping (GROUP BY on dimension)
────────────────────────────────────────────────────
duckdb bench_sf01.db -c ".timer on" \
  "SELECT season, COUNT(*) FROM photos WHERE season IS NOT NULL GROUP BY season;"

Expected performance:
  SF 0.1:  < 5 ms
  SF 1.0:  5-20 ms
  SF 5.0:  20-100 ms
  SF 10.0: 50-250 ms

BENCHMARK QUERY 3: Multi-dimensional aggregation
─────────────────────────────────────────────────
duckdb bench_sf01.db -c ".timer on" \
  "SELECT make, model, season, COUNT(*) as cnt, AVG(iso) 
   FROM photos 
   WHERE make IS NOT NULL AND season IS NOT NULL 
   GROUP BY make, model, season 
   ORDER BY cnt DESC 
   LIMIT 100;"

Expected performance:
  SF 0.1:  < 20 ms
  SF 1.0:  20-100 ms
  SF 5.0:  100-500 ms
  SF 10.0: 300-1500 ms

BENCHMARK QUERY 4: Nested aggregation (OLAP roll-up)
──────────────────────────────────────────────────────
duckdb bench_sf01.db -c ".timer on" \
  "SELECT 
       make, 
       COUNT(*) as photos,
       COUNT(DISTINCT CASE WHEN iso > 3200 THEN filename END) as high_iso_count
   FROM photos 
   WHERE make IS NOT NULL 
   GROUP BY make 
   ORDER BY photos DESC LIMIT 20;"

Expected performance:
  SF 0.1:  < 10 ms
  SF 1.0:  10-50 ms
  SF 5.0:  50-300 ms
  SF 10.0: 200-1000 ms

BENCHMARK QUERY 5: String operations (category explosion)
───────────────────────────────────────────────────────────
duckdb bench_sf01.db -c ".timer on" \
  "SELECT 
       cat,
       COUNT(*) as cnt
   FROM (
       SELECT UNNEST(categories) as cat FROM photos
   )
   GROUP BY cat
   ORDER BY cnt DESC;"

Expected performance:
  SF 0.1:  < 30 ms
  SF 1.0:  30-150 ms
  SF 5.0:  150-750 ms
  SF 10.0: 500-2500 ms


CROSS-PLATFORM BENCHMARKING:
──────────────────────────────

Compare performance: Windows 11 vs Linux

Setup:
  1. Same photo directory on both systems (or similar size)
  2. Run with same --max-files limit
  3. Measure both: Processing time + Query time

Test Script (benchmark.sh for Linux):
─────────────────────────────────────
#!/bin/bash

echo "=== Scale Factor 0.1 (100 files) ==="
time python photo_analyze_jpg_only.py --root-dir ./photos --max-files 100 \
    --db-path bench_test.db

echo ""
echo "=== Query Performance Test ==="
time duckdb bench_test.db -c "SELECT season, COUNT(*) FROM photos GROUP BY season;"

rm bench_test.db


Test Script (benchmark.ps1 for Windows PowerShell):
───────────────────────────────────────────────────
Write-Host "=== Scale Factor 0.1 (100 files) ==="
$sw = [Diagnostics.Stopwatch]::StartNew()
python photo_analyze_jpg_only.py --root-dir C:\photos --max-files 100 --db-path bench_test.db
$sw.Stop()
Write-Host "Elapsed: $($sw.ElapsedMilliseconds) ms"

Write-Host ""
Write-Host "=== Query Performance Test ==="
$sw = [Diagnostics.Stopwatch]::StartNew()
duckdb bench_test.db -c "SELECT season, COUNT(*) FROM photos GROUP BY season;"
$sw.Stop()
Write-Host "Elapsed: $($sw.ElapsedMilliseconds) ms"

Remove-Item bench_test.db


EXPECTED RESULTS:
─────────────────

Linux (SSD, i7+):
  Processing 100 images:  2-3 min
  SF 1.0 query:           20-50 ms
  SF 10.0 query:          200-500 ms

Windows 11 (SSD, i7+):
  Processing 100 images:  3-5 min (antivirus overhead)
  SF 1.0 query:           30-80 ms
  SF 10.0 query:          300-800 ms

With GPU (CLIP):
  Processing 100 images:  30-60 sec
  SF 1.0 query:           (same, ~20-50 ms)


================================================================================
PART 3: PYTHON BENCHMARKING HARNESS
================================================================================

Create a Python script for automated benchmarking:

```python
# benchmark_harness.py
import subprocess
import time
import duckdb
from pathlib import Path

SCALE_FACTORS = {
    "sf01": 100,
    "sf10": 1000,
    "sf50": 5000,
}

QUERIES = {
    "count_all": "SELECT COUNT(*) FROM photos;",
    "by_season": "SELECT season, COUNT(*) FROM photos WHERE season IS NOT NULL GROUP BY season;",
    "by_camera": "SELECT make, COUNT(*) FROM photos WHERE make IS NOT NULL GROUP BY make;",
}

def benchmark_processing(root_dir, sf_name, max_files):
    """Benchmark photo processing."""
    db_path = f"bench_{sf_name}.db"
    
    start = time.time()
    result = subprocess.run([
        "python", "photo_analyze_jpg_only.py",
        "--root-dir", root_dir,
        "--max-files", str(max_files),
        "--db-path", db_path
    ], capture_output=True, text=True)
    elapsed = time.time() - start
    
    print(f"{sf_name}: {elapsed:.2f} seconds")
    return db_path, elapsed

def benchmark_queries(db_path):
    """Benchmark query execution."""
    conn = duckdb.connect(db_path, read_only=True)
    
    for query_name, query in QUERIES.items():
        start = time.time()
        result = conn.execute(query).fetchall()
        elapsed = time.time() - start
        
        print(f"  {query_name}: {elapsed*1000:.2f} ms (rows: {len(result)})")
    
    conn.close()

if __name__ == "__main__":
    root_dir = "./photos"
    
    results = {}
    for sf_name, max_files in SCALE_FACTORS.items():
        print(f"\n=== {sf_name} ({max_files} files) ===")
        db_path, proc_time = benchmark_processing(root_dir, sf_name, max_files)
        results[sf_name] = {"processing": proc_time}
        benchmark_queries(db_path)
    
    print("\n=== SUMMARY ===")
    for sf_name, times in results.items():
        print(f"{sf_name}: {times['processing']:.2f} sec processing")
```

Run it:
  python benchmark_harness.py

================================================================================
PART 4: MONITORING & METRICS
================================================================================

Track important metrics across scale factors:

SQL to export metrics:
──────────────────────
SELECT 
    (SELECT COUNT(*) FROM photos) as total_photos,
    (SELECT COUNT(DISTINCT make) FROM photos WHERE make IS NOT NULL) as unique_makes,
    (SELECT COUNT(DISTINCT category) FROM (SELECT UNNEST(categories) as category FROM photos)) as unique_categories,
    (SELECT COUNT(*) FROM photos WHERE gps_lat IS NOT NULL) as gps_tagged,
    (SELECT COUNT(*) FROM photos WHERE datetime_original IS NOT NULL) as datetime_tagged,
    (SELECT AVG(image_width * image_height) FROM photos) as avg_megapixels,
    (SELECT MAX(datetime_original) FROM photos WHERE datetime_original IS NOT NULL) as latest_photo,
    (SELECT MIN(datetime_original) FROM photos WHERE datetime_original IS NOT NULL) as earliest_photo
;

This gives you:
  ✓ Coverage metrics (% with GPS, datetime, etc.)
  ✓ Data quality indicators
  ✓ Temporal range of collection
  ✓ Average media quality

================================================================================
SUMMARY: BENCHMARKING CHECKLIST
================================================================================

Before benchmark:
  ☐ Ensure consistent file source across platforms
  ☐ Close other applications to minimize interference
  ☐ On Windows: Temporarily disable antivirus if possible
  ☐ Use SSD for consistent I/O performance
  ☐ Pin CPU frequency if available (avoid thermal throttling)

During benchmark:
  ☐ Record processing time (--max-files limit)
  ☐ Record query times (duckdb with .timer on)
  ☐ Document system specs (CPU, RAM, GPU, OS)
  ☐ Note any errors or warnings
  ☐ Take 2-3 runs per test (report average)

Results to capture:
  ☐ SF 0.1, 1.0, 5.0, 10.0 processing times
  ☐ Query times per scale factor
  ☐ Memory usage peaks
  ☐ Linux vs Windows comparison
  ☐ GPU vs CPU performance (if testing both)

Analysis:
  ☐ Plot query time vs scale factor
  ☐ Calculate performance ratios (2x files = 2x time?)
  ☐ Identify bottlenecks (I/O? ML inference? DuckDB?)
  ☐ Compare against TPC-DS reference numbers

================================================================================
