# protein

Scripts associated with my CSCE 790: Deep Reinforcement Learning and Search final project.

## Prerequisites

To run these scripts, you will need to have installed `git`, `conda`, and a modified version of [REDCRAFT](https://redcraft.readthedocs.io/en/latest/) on your system.

### Installing REDCRAFT

These instructions are for Unix-based targets, such as Linux distributions, macOS, and WSL2 environments. If you are using Windows, you are on your own.

1. Clone the REDCRAFT repository within your local directory of choice using the following command:

    ```bash
    git clone https://bitbucket.org/hvalafar/redcraft.git
    ```

2. `cd` into the `redcraft/` directory now on your system. Here we need to do two things. First, fetch the modified version of `molan.cpp` using this command:

    ```bash
    wget https://raw.githubusercontent.com/singhish/RLProteinFolding/main/molan.cpp
    ```

    and replace `src/molan.cpp` with this file. Next, replace the empty `googletest/` directory by running:

    ```bash
    rm -r googletest/
    git clone https://github.com/google/googletest.git
    ```

3. You are now free to follow the instructions located [here](https://redcraft.readthedocs.io/en/latest/usage/installation.html). If you are using WSL2, before running `make`, run the following command:

    ```bash
    sudo strip --remove-section=.note.ABI-tag /usr/lib/x86_64-linux-gnu/libQt5Core.so.5
    ```

4. Test your REDCRAFT installation by running `redcraft`. You should get the following output if the installation was successful:

    ```bash
    $ redcraft
    Usage: redcraft [-S|--script] <binary|script> [options] [-V|--version]
    See redcraft --help or the REDCRAFT documentation for details
    ```

### Conda Environment Setup

To install and activate the prerequisite Python modules for this project, run:

```bash
conda env create -f environment.yml -n protein
conda activate protein
```
