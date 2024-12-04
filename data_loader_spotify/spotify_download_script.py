import os

# Function to download a Spotify playlist
def download_spotify_tracks(track_url, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Construct the command to download the playlist and save it to the specified folder
    command = f"spotdl {track_url} --format wav --output \"{output_folder}"
    
    # Execute the command
    os.system(command)

# Take user input for the playlist URL



if __name__ == "__main__":
    # Download the playlist
    playlist_url = "https://open.spotify.com/track/53QF56cjZA9RTuuMZDrSA6"
    output_folder = "Downloaded_Songs"
    download_spotify_tracks(playlist_url, output_folder)
    