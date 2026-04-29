"""Generate HTML documentation using pdoc."""
import subprocess
import sys


def main():
    """Run pdoc to generate HTML docs for the package."""
    subprocess.run([
    sys.executable, "-m", "pdoc", 
    "-o", "html", 
    "crypto_momentum_lab"
], check=True)
    print("Docs generated in docs/html/")


if __name__ == "__main__":
    main()
