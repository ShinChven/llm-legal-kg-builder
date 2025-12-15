-- Combined Centrality Metrics 📈
-- To get a complete view, you can combine these two queries to list all unique nodes
-- and their corresponding in-degree and out-degree counts.
-- This gives a comprehensive picture of each node's role in the network.

WITH out_degrees AS (
    SELECT
        subject_name AS node,
        COUNT(*) AS out_degree
    FROM
        act_relationships
    GROUP BY
        subject_name
),
in_degrees AS (
    SELECT
        object_name AS node,
        COUNT(*) AS in_degree
    FROM
        act_relationships
    GROUP BY
        object_name
)
SELECT
    COALESCE(o.node, i.node) AS node,
    COALESCE(o.out_degree, 0) AS out_degree,
    COALESCE(i.in_degree, 0) AS in_degree
FROM
    out_degrees o
FULL OUTER JOIN
    in_degrees i ON o.node = i.node
ORDER BY
    node;


-- Out-Degree Centrality
-- The out-degree of a node is the number of relationships it initiates (subject_name).
-- A node with a high out-degree is very active or influential in starting connections.
-- The following query calculates the out-degree for every subject:

SELECT
    subject_name,
    COUNT(*) AS out_degree
FROM
    act_relationships
GROUP BY
    subject_name
ORDER BY
    out_degree DESC;


-- In-Degree Centrality
-- The in-degree of a node is the number of relationships it receives (object_name).
-- A node with a high in-degree is popular or a hub that receives many connections from others.
-- The following query calculates the in-degree for every object:

SELECT
    object_name,
    COUNT(*) AS in_degree
FROM
    act_relationships
GROUP BY
    object_name
ORDER BY
    in_degree DESC;


select DISTINCT subject_name, count(*) from act_relationships where subject_name ILIKE 'legislation act%' group by subject_name;
select DISTINCT object_name, count(*) from act_relationships where object_name ILIKE 'legislation act%' group by object_name;


-- Count how many legislations mention each "Legislation Act YYYY"
-- and how many total mentions appear across all texts.
WITH mentions AS (
	SELECT
		l.id AS legislation_id,
		(regexp_matches(l.text, '(Legislation[[:space:]]+Act[[:space:]]+[0-9]{4})', 'gi'))[1] AS act
	FROM legislations l
	WHERE l.source = 'legislation.govt.nz'
)
SELECT
	lower(act) AS act,
	COUNT(DISTINCT legislation_id) AS documents_referencing,
	COUNT(*) AS total_mentions
FROM mentions
GROUP BY lower(act)
ORDER BY total_mentions DESC;


WITH mentions AS (
    -- Extract all mentions of 'Legislation Act YYYY' from the text of all legislations
    SELECT
        l.year AS doc_year, -- The year of the document that contains the mention
        (regexp_matches(l.text, '(Legislation Act [0-9]{4})', 'gi'))[1] AS act_mention
    FROM
        legislations l
    WHERE
        l.source = 'legislation.govt.nz' AND l.text ~ 'Legislation Act [0-9]{4}'
),
parsed_mentions AS (
    -- Parse the year from the mentioned act string
    SELECT
        doc_year,
        act_mention,
        (regexp_matches(act_mention, '([0-9]{4})'))[1]::int AS act_year
    FROM
        mentions
)
-- Count how many older documents reference each of the target acts
SELECT
    act_mention AS target_act,
    COUNT(*) AS referencing_acts_before
FROM
    parsed_mentions
WHERE
    act_mention IN ('Legislation Act 2012', 'Legislation Act 2019')
    AND doc_year < act_year
GROUP BY
    act_mention
ORDER BY
    act_mention;


select count(*) from legislations where text ilike '%Legislation Act 2012%' and year < 2012;
select count(*) from legislations where text ilike '%Legislation Act 2019%' and year < 2019;
