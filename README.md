# UniVR-AI

UniVR-AI is a repository containing the code of the artificial intelligence course of the University of Verona. This project includes various modules for search algorithms, Markov decision processes, machine learning, and reinforcement learning.

## Project Structure

The project is structured as follows:

- `data/`: project data:
  - `dnn/`: data for deep neural networks;
- `envs/`: environments for reinforcement learning;
  - `collections/`: collections of environments types;
- `inc/`: includes various support modules:
  - `ai/`: artificial intelligence functions;
  - `collections/`: collection of types;
  - `constants/`: constants used in the project;
  - `graphics/`: graphics functions;
  - `types/`: data types used in the project;
  - `utils/`: various utilities;
- `src/`: main source code:
  - `machine_learning/`: machine learning algorithms;
  - `markov_decision_processes/`: Markov decision processes algorithms;
  - `reinforcement_learning/`: reinforcement learning algorithms;
  - `search/`: search algorithms;
- `tests/`: tests for various modules.

## Requirements

To install the project requirements, run:

```bash
pip install -r requirements.txt
```

## Setup

To setup the project, set the following environment variables:

- `TF_CPP_MIN_LOG_LEVEL = 1`: to disable TensorFlow warnings;
- `TF_ENABLE_ONEDNN_OPTS = 1`: to enable oneDNN optimizations.

## Usage

In test folder you can find some examples of how to use the modules of the project.

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.

## Contributing

Contributions are welcome! Please open an issue or a pull request to discuss any changes.

## Authors

- [Lecini Rustem](https://github.com/RustemL02)
