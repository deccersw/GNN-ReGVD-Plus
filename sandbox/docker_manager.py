"""
Docker Manager — builds sandbox images and manages container lifecycle.

Handles:
  - Building the sandbox Docker image from Dockerfile
  - Checking if the image exists
  - Container cleanup
"""

import os
import logging
import subprocess

logger = logging.getLogger(__name__)

DOCKERFILES_DIR = os.path.join(os.path.dirname(__file__), "dockerfiles")
DEFAULT_IMAGE_NAME = "gnn-regvd-sandbox"
DEFAULT_IMAGE_TAG = "c-cpp"


def get_image_name(tag: str = DEFAULT_IMAGE_TAG) -> str:
    return f"{DEFAULT_IMAGE_NAME}:{tag}"


def image_exists(tag: str = DEFAULT_IMAGE_TAG) -> bool:
    name = get_image_name(tag)
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", name],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def build_image(tag: str = DEFAULT_IMAGE_TAG,
                dockerfile: str = "Dockerfile.c_cpp",
                force: bool = False) -> bool:
    """
    Build the sandbox Docker image.

    Returns True on success.
    """
    name = get_image_name(tag)

    if not force and image_exists(tag):
        logger.info(f"Image {name} already exists, skipping build")
        return True

    dockerfile_path = os.path.join(DOCKERFILES_DIR, dockerfile)
    if not os.path.exists(dockerfile_path):
        logger.error(f"Dockerfile not found: {dockerfile_path}")
        return False

    logger.info(f"Building Docker image {name} from {dockerfile_path}...")
    try:
        result = subprocess.run(
            ["docker", "build", "-t", name, "-f", dockerfile_path, DOCKERFILES_DIR],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            logger.info(f"Successfully built {name}")
            return True
        logger.error(f"Docker build failed:\n{result.stderr}")
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.error(f"Docker build error: {e}")
        return False


def cleanup_containers(label: str = "gnn-regvd-sandbox"):
    """Remove all stopped containers with our label."""
    try:
        subprocess.run(
            ["docker", "container", "prune", "-f",
             "--filter", f"label={label}"],
            capture_output=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def get_seccomp_profile_path() -> str:
    return os.path.join(DOCKERFILES_DIR, "seccomp_sandbox.json")
