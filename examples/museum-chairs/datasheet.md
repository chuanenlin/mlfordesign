# Datasheet: museum chairs

Answers to the questions in Gebru et al., "Datasheets for Datasets" (CACM 2021), for the example set used in weeks 4 and 5 of Machine Learning for Design.

## Motivation

- **Purpose:** a set of designs that every student can see, map, fine-tune on and share, so the course can show each step on the same images before students use their own.
- **Created by:** the course instructor, from two museums' open-access collections. No funding.

## Composition

- **Instances:** 56 photographs, one chair in each, on a plain gray or white backdrop.
- **Sources:** 28 from The Metropolitan Museum of Art and 28 from the Art Institute of Chicago.
- **Sample, not the whole collection:** I looked at all 36 public-domain chairs with images at the Art Institute and about 330 chairs at the Met, and picked by hand across dates, makers and regions.
- **Each image has:** the museum, object number, title, maker, date, materials, credit line, license and a link to the museum's record (credits.csv).
- **Missing:** chairs made after 1904, because most later chairs are not public domain, and most chairs made outside Europe and America. Only four chairs are from outside Europe and America: three were made for European buyers (one in India, one in China, and one listed as the West Indies or Asia), and one is from Africa.
- **People:** no people appear in the photographs.

## Collection process

- Records came from the Met's Collection API and the Art Institute's API, and images from each museum's own image server.
- Left out: black-and-white photos, colored backdrops, photos with more than one chair, and the second chair of a pair.

## Preprocessing

- The originals are unchanged. The training copies are resized so the long side is 1,024 px.
- Captions: captions-museum.csv has one caption for each chair, written from its museum record. In the week 5 example, one model is trained without these captions, and the trainer writes a caption for each image with a captioning model.

## Uses

- Mapping with CLIP, fine-tuning an image model and testing concept sliders (e.g., making a chair more or less ornate) in this course.
- Not suited to judging a chair's age, maker or value, since the set is small and picked by hand.

## Distribution

- All 56 images are CC0 (public domain). credits.csv keeps each museum's credit line so every image can be traced to its record.

## Maintenance

- The course instructor keeps the set and its records together in this repository.
