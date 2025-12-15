-- This script identifies legislations (Acts) that have no recorded relationships.
-- It checks for Acts in the 'legislations' table that do not appear in the 'act_relationships' table,
-- either as a subject or an object.
-- The query specifically excludes Acts where the 'word_count' is NULL,
-- and it returns the title, year, word count, and PDF link for these isolated Acts.
-- The results are ordered by word count in descending order to prioritize larger documents.

SELECT
    l.title,
		l.year,
		l.word_count,
		l.pdf_link,
    COUNT(ar.id) AS total_connections
FROM
    legislations l
LEFT JOIN
    act_relationships ar ON l.title = ar.subject_name OR l.title = ar.object_name
WHERE
    l.word_count IS NOT NULL
GROUP BY
    l.title, l.year, l.word_count, l.pdf_link
HAVING
    COUNT(ar.id) = 0
ORDER BY
    l.word_count DESC;


select title, word_count, pdf_link from legislations order by word_count DESC;
