SELECT
    COUNT(*) FILTER (
        WHERE
            similarity > 0.99
    ) AS total_matches,
    COUNT(*) FILTER (
        WHERE
            similarity IS NULL
    ) AS similarity_null,
    COUNT(*) AS total_mismatch,
    (
        SELECT COUNT(*)
        FROM act_relationships
    ) AS total_relationships,
    ROUND(
        COUNT(*) FILTER (
            WHERE
                similarity > 0.99
        )::numeric / NULLIF(COUNT(*), 0) * 100,
        2
    ) AS P99_matches_percentage,
    ROUND(
        COUNT(*) FILTER (
            WHERE
                similarity > 0.99
        )::numeric / NULLIF(
            (
                SELECT COUNT(*)
                FROM act_relationships
            ),
            0
        ) * 100,
        2
    ) AS fix_rate_percentage
FROM lost_and_found;

SELECT
-- laf.object_name,
ar.object_name AS matched_object_name,
laf.found_title,
laf.similarity
FROM
    lost_and_found laf
    JOIN act_relationships ar ON ar.object_name = laf.object_name
WHERE
    laf.similarity > 0.99;

-- All relationship fixed using similarity search

select
    object_name,
    found_title,
    similarity
from lost_and_found
where
    similarity > 0.99;

select
    object_name,
    found_title,
    similarity
from lost_and_found
where
    similarity < 0.99
    and similarity > 0.98;

-- Example of fixing title from different language https://www.legislation.govt.nz/act/local/1915/0001/latest/DLM40145.html
select
    object_name,
    found_title,
    similarity
from lost_and_found
where
    similarity > 0.99
    and object_name ilike '%plenty%';
