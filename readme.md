This project is in rapid active development, I apologise if things don't work/run.

Feel free to submit a bug report or suggestions by stopping by my desk or
messaging me on Slack.

Python Code is in pysrc/ismatching

To build and install, run ```pip install .``` within the ismatching-lib
directory.

This package assumes that the following python packages are installed (Note
that this list is subject to change):
- pymatching
- numpy
- scipy


TODO:
- Examples
- Documentation
- Make Code Work Good
- Organise Code Properly
- Document internals
  - Python
  - Rust
- Use bit packing to save on memory
- Accept Stim Circuits
  - Possible lead: https://github.com/Strilanc/honeycomb_threshold/blob/main/src/noise.py
