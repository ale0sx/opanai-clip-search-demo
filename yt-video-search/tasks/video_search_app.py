from tempfile import TemporaryDirectory
from typing import Tuple, List, Any

import cv2
import streamlit as st
import torch
import youtube_dl

from PIL import Image

from image_embeddings import compute_image_embeddings
from text_to_image_search import text_to_image_search
from image_to_image_search import image_to_image_search

st.set_page_config(
    page_title="Search Videos for Things",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache
def download_video_cached(
    video_url: str, frame_frequency: int = 1
) -> List[Image.Image]:
    """
    Download video from Youtube and return frames from it sampled every frame_frequency
    seconds

    Args:
        video_url: URL link to a Youtube video
        frame_frequency: How often to sample fromes from the video, as a rate in seconds

    Returns:
        List[Image.Image]: List of displayable images from the video
    """
    with TemporaryDirectory() as download_directory:
        # Download the video as 'video.mp4' in the temporary directory
        # See: https://stackoverflow.com/a/63002071 for how to get more info
        youtube_dl_options = {
            "outtmpl": f"{download_directory}/video.mp4",
            "extractaudio": False,
        }

        # Perform the download
        with youtube_dl.YoutubeDL(youtube_dl_options) as ydl:
            ydl.download([video_url])

        # Iterate through video, extracting a frame every second
        path_to_video = f"{download_directory}/video.mp4"
        video = cv2.VideoCapture(path_to_video)

        # Get frames per second to load a frame every second
        fps = video.get(cv2.CAP_PROP_FPS)
        multiplier = int(fps * frame_frequency)

        images = []

        while True:
            success, frame = video.read()

            if not success:
                break

            frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))

            if frame_number % multiplier == 0:
                images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        video.release()

    return images


@st.cache(hash_funcs={torch.Tensor: id})
def compute_image_embeddings_cached(
    images: List[Image.Image]
) -> torch.Tensor:
    """ Cached version of compute_image_embeddings() """
    return compute_image_embeddings(images=images)


@st.cache(hash_funcs={torch.Tensor: id})
def text_to_image_search_cached(
    search_query: str,
    list_of_images: List[Image.Image],
    image_features: torch.Tensor,
    top_k: int = 1,
) -> Tuple[List[Image.Image], List[int], List[float]]:
    """ Cached version of text_to_image_search() """
    return text_to_image_search(
        search_query=search_query,
        list_of_images=list_of_images,
        image_features=image_features,
        top_k=top_k,
    )


@st.cache(hash_funcs={torch.Tensor: id})
def image_to_image_search_cached(
    search_query: Image.Image,
    list_of_images: List[Image.Image],
    image_features: torch.Tensor,
    top_k: int = 1,
) -> Tuple[List[Image.Image], List[int], List[float]]:
    """ Cached version of text_to_image_search() """
    return image_to_image_search(
        search_query=search_query,
        list_of_images=list_of_images,
        image_features=image_features,
        top_k=top_k,
    )


@st.cache
def load_image_cached(image_file: Any) -> Image.Image:
    """ Cached function to load uploaded image file as PIL.Image.Image """
    img = Image.open(image_file)
    return img


# Initialize queries to None so if statements don't break
image_query = None
text_query = None

# Header and sidebar
st.title("Search for stuff in Youtube videos")
sample_frequency = st.sidebar.selectbox(
    label="Sample every how many seconds?",
    options=[2, 1, 0.5],
    help="Smaller numbers will increase runtime",
)
number_matches_to_show = st.sidebar.slider(
    label="How many matches to show", min_value=1, max_value=10, value=3
)

# Initial inputs - video link from Youtube and whether to search with text or an image
video_to_download = st.text_input(label="Paste a link to a Youtube video")
search_type = st.selectbox(
    label="Choose to search by entering text or uploading an image",
    options=["Text", "Image"],
    help="This is like doing a normal Google search, " "or a Google image search.",
)

if video_to_download:

    # Format the correct link URL depending on the video link type
    if "youtu.be" in video_to_download:
        timestamp_link = f"{video_to_download}?t="
    else:
        timestamp_link = f"{video_to_download}&t="

    # Display the video
    st.video(video_to_download)

    # Process video frames and create CLIP embeddings
    images_from_video = download_video_cached(
        video_url=video_to_download, frame_frequency=sample_frequency
    )
    image_embeddings = compute_image_embeddings_cached(images_from_video)

    # Prompt user for what to search for (text vs. image)
    if search_type == "Text":
        text_query = st.text_input(
            label="Type what you're searching for (e.g. a cat)",
            help="It can be complex too! E.g. " "a cat sitting wearing a hoodie",
        )
    else:
        image_query = st.file_uploader(
            label="Upload an image for your search",
            type=["jpg", "jpeg", "png"],
            help="This searches the video for frames similar to your uploaded image"
        )

    # Do text search
    if text_query:
        matching_images, matching_indices, probabilities = text_to_image_search_cached(
            search_query=text_query,
            list_of_images=images_from_video,
            image_features=image_embeddings,
            top_k=number_matches_to_show,
        )

        for image, index, _ in zip(matching_images, matching_indices, probabilities):
            st.write(
                f"Video time for this match: {timestamp_link}{index * sample_frequency}s"
            )
            st.image(image)

    # Do image search
    elif image_query:
        # Display the image you're searching for
        st.image(image_query)
        st.write("Searching parts of the Youtube video similar to the above uploaded "
                 "image")
        st.header("Results:")

        image_query = load_image_cached(image_file=image_query)
        matching_images, matching_indices, probabilities = image_to_image_search_cached(
            search_query=image_query,
            list_of_images=images_from_video,
            image_features=image_embeddings,
            top_k=number_matches_to_show,
        )

        for image, index, _ in zip(matching_images, matching_indices, probabilities):
            st.write(
                f"Video time for this match: {timestamp_link}{index * sample_frequency}s"
            )
            st.image(image)
