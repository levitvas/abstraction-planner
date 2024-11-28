import cProfile
import pstats
from pstats import SortKey

from planner import create_plan

# Method 1: Profile the whole program
python_command = """
import cProfile
cProfile.run('create_plan()', 'stats.prof')
"""


# Method 2: Profile specific parts (probably better for your case)
def main():
    profiler = cProfile.Profile()
    profiler.enable()

    # Your planning app code here
    create_plan()

    profiler.disable()
    # Sort by cumulative time and print top 20 functions
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)
    # Optionally save to file
    stats.dump_stats("program_profile.prof")

if __name__ == "__main__":
    main()
