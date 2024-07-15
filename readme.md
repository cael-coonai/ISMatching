This project is in rapid active development, I apologise if things don't work/run.

Feel free to submit a bug report or suggestions by stopping by my desk or
messaging me on Slack.

Python Code is in pysrc/ismatching

To build, run ```maturin build --release``` in this directory then run 
```pip install``` on the generated wheel.

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
- Rename function parameters
- Document external api
- Document internals
  - Python
  - Rust
- Use bit packing to save on memory
- Accept Stim Circuits
