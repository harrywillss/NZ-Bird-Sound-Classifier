import os
import json
import requests
import pandas as pd

# Using advanced search to find bird songs in New Zealand
# see https://xeno-canto.org/help/search

# Bird types: ["Fantail", "Kākā", "Bellbird", "Tūī", "Kea", "Morepork", "Kākāpō", "House Sparrow"], # List of bird types to search for
# Recording types: ["song", "call"]

class BirdRecordingDownloader:
    BASE_URL = "https://xeno-canto.org/api/3/recordings"

    def __init__(self, country="=New Zealand", group="birds", recording_type="song", quality=">C", output_dir="downloads"):
        self.country = country
        self.group = group
        self.recording_type = recording_type
        self.quality = quality
        self.output_dir = output_dir
        self.api_key = os.getenv("XC_API_KEY")

        if not self.api_key:
            raise ValueError("API key not found. Please set the XC_API_KEY environment variable.")

        os.makedirs(self.output_dir, exist_ok=True)
        self.query_string = self.build_query()

    def build_query(self):
        return f'cnt:"{self.country}" grp:"{self.group}" type:"{self.recording_type}" q:"{self.quality}"'

    def fetch_data(self):
        print("🔍 Fetching data with query:", self.query_string)
        all_recordings = []
        page = 1
        while True:
            response = requests.get(
                self.BASE_URL,
                params={"query": self.query_string, "key": self.api_key, "page": page}
            )
            if response.status_code != 200:
                raise Exception(f"Failed to fetch data: {response.status_code} {response.text}")
            data = response.json()
            if page == 1:
                print(f"✅ Data fetched successfully. Recordings: {data['numRecordings']}, Species: {data['numSpecies']}")
            all_recordings.extend(data.get("recordings", []))
            if page >= int(data.get("numPages", 1)):
                break
            page += 1
        return {"recordings": all_recordings}

    def download_recordings(self, recordings):
        downloaded = []
        for rec in recordings:
            if "file" not in rec or "id" not in rec:
                continue

            file_url = rec["file"]
            file_id = rec["id"]
            gen = rec.get("gen", "Unknown")
            sp = rec.get("sp", "Unknown")

            filename = f"{file_id}_{gen}_{sp}.mp3"
            filepath = os.path.join(self.output_dir, filename)

            print(f"⬇️ Downloading {filename}...")
            resp = requests.get(file_url)
            if resp.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                print(f"✅ Saved: {filename}")
                downloaded.append(filepath)
            else:
                print(f"❌ Failed to download {file_id}: {resp.status_code}")
        return downloaded

    def save_metadata(self, recordings, metadata_filename="recordings_metadata.json", csv_filename="recordings_data.csv"):
        metadata = {
            "query": self.query_string,
            "total_recordings": len(recordings),
            "recordings": [
                {
                    "id": rec["id"],
                    "generic_name": rec.get("gen", "Unknown"),
                    "scientific_name": rec.get("sp", "Unknown"),
                    "english_name": rec.get("en", "Unknown"),
                    "sex": rec.get("sex", "Unknown"),
                    "file_url": rec["file"],
                    "sonogram_url": rec.get("sono", None),
                    "oscillogram_url": rec.get("osci", None),
                    "length": rec.get("length", "Unknown")
                }
                for rec in recordings if "file" in rec and "id" in rec
            ]
        }

        with open(metadata_filename, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"📁 Metadata saved to {metadata_filename}")

        df = pd.DataFrame(metadata["recordings"])
        df.to_csv(csv_filename, index=False)
        print(f"📊 CSV data saved to {csv_filename}")

    def report_summary(self, recordings):
        if not recordings:
            print("⚠️ No recordings to summarize.")
            return

        df = pd.DataFrame([
            {
                "species": f"{rec.get('gen', 'Unknown')} {rec.get('sp', 'Unknown')}",
                "english_name": rec.get("en", "Unknown"),
                "length": self._parse_length(rec.get("length", "0:00")),
                "sex": rec.get("sex", "Unknown")
            }
            for rec in recordings if "file" in rec and "id" in rec
        ])

        print("\n📋 --- Recordings Summary ---")
        print(f"Total recordings: {len(df)}")
        print(f"Unique species: {df['species'].nunique()}")

        print("\nTop 10 species by number of recordings:")
        print(df['species'].value_counts().head(10))

        avg_length = df['length'].mean()
        print(f"\nAverage recording length: {avg_length:.2f} seconds")

        print("\nRecordings by sex:")
        print(df['sex'].value_counts())

    def _parse_length(self, length_str):
        # Converts "m:ss" to seconds
        try:
            mins, secs = map(int, length_str.split(":"))
            return mins * 60 + secs
        except Exception:
            return 0

    def run(self):
        data = self.fetch_data()
        recordings = data.get("recordings", [])
        if not recordings:
            print("⚠️ No recordings found.")
            return

        #self.download_recordings(recordings)
        #self.save_metadata(recordings)
        self.report_summary(recordings)


if __name__ == "__main__":
    downloader = BirdRecordingDownloader()
    downloader.run()