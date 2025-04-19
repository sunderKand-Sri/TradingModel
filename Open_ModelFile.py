import os
import pickle
from pathlib import Path


def load_and_inspect_pkl(filepath):
    """Load a .pkl file and return its contents."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading {filepath}: {str(e)}")
        return None


def main():
    # Set directory path
    pkl_dir = Path("C:/Users/Windows/Desktop/modelresult")

    # Get all .pkl files in directory
    pkl_files = list(pkl_dir.glob('*.pkl'))

    if not pkl_files:
        print("No .pkl files found in the directory.")
        return

    print(f"Found {len(pkl_files)} .pkl files to process:")
    for i, file in enumerate(pkl_files, 1):
        print(f"{i}. {file.name}")

    # Process each file with user confirmation
    for file in pkl_files:
        print(f"\n{'=' * 50}")
        print(f"Processing: {file.name}")
        print(f"{'=' * 50}")

        # Load the file
        data = load_and_inspect_pkl(file)

        if data is not None:
            # Display basic info about the loaded object
            print(f"\n📦 File Contents:")
            print(f"Type: {type(data)}")

            # Special handling for common model types
            if hasattr(data, '__class__'):
                print(f"Class: {data.__class__.__name__}")

            if isinstance(data, dict):
                print(f"Keys: {list(data.keys())}")
                print("\nSample contents:")
                for k, v in list(data.items())[:3]:
                    print(f"{k}: {str(v)[:100]}...")

            elif hasattr(data, 'get_params'):  # For scikit-learn models
                print("\nModel Parameters:")
                print(data.get_params())

            print(f"\nObject summary:\n{data}")

        # Prompt user to continue
        while True:
            user_input = input("\nContinue to next file? (y/n): ").lower()
            if user_input in ('y', 'yes'):
                break
            elif user_input in ('n', 'no'):
                print("Exiting...")
                return
            else:
                print("Please enter 'y' or 'n'")


if __name__ == "__main__":
    main()