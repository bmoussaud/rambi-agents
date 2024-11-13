from dataclasses import dataclass
from tmdbv3api import TMDb, Search
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Movie:
    title: str
    plot: str
    posterUrl: str


def get_movie_info(title) -> Movie:
    search_results = Search().movies(title)
    if search_results:
        sr = search_results[0]  # Return the first result
        return Movie(sr.title, sr.overview, f"https://image.tmdb.org/t/p/original/{sr.poster_path}")
    else:
        return None


if __name__ == "__main__":
    movie_title = 'Bambi'
    movie_info = get_movie_info(movie_title)
    if movie_info:
        print(movie_info)
    else:
        print("Movie not found.")
