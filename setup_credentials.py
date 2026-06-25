"""
setup_credentials.py — A helper script to configure your Kaggle API key locally.
This script creates a .env file so you don't have to share your key publicly.
"""
import os

def setup():
    print("=" * 60)
    print("  🌿 PLANT LEAF CLASSIFIER - Kaggle API Setup")
    print("=" * 60)
    print("\n[INFO] This script will create a '.env' file in your project directory.")
    print("[INFO] You can get your key from: https://www.kaggle.com/settings (API -> Create New Token)\n")

    username = input("  Enter your Kaggle Username: ").strip()
    key = input("  Enter your Kaggle API Key: ").strip()

    if not username or not key:
        print("\n[ERROR] Username and Key cannot be empty!")
        return

    env_content = f"KAGGLE_USERNAME={username}\nKAGGLE_KEY={key}\n"
    
    with open(".env", "w") as f:
        f.write(env_content)

    # Also create a .gitignore to make sure .env is not committed if you use Git
    if not os.path.exists(".gitignore"):
        with open(".gitignore", "w") as f:
            f.write(".env\n__pycache__/\ndataset/\nmodels/*.keras\nresults/*.png\n")

    print("\n" + "=" * 60)
    print("  [SUCCESS] Credentials saved to .env file!")
    print("  [INFO] You can now run 'python download_dataset.py'.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    setup()
