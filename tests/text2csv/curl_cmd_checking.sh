curl -X 'POST' \
  'http://localhost:8090/v1/text2csv?input_text=MATCH%20%28%3AMovie%20%7Btitle%3A%20%27Casino%27%7D%29%3C-%5B%3AACTED_IN%5D-%28actor%3APerson%29%20RETURN%20actor.name%20AS%20actor' \
  -H 'accept: application/json' \
  -d ''
