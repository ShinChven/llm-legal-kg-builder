SELECT DISTINCT
    subject_name
FROM act_relationships
WHERE
    subject_name NOT IN (
        SELECT title
        FROM legislations
    );

DELETE FROM act_relationships
WHERE
    subject_name IN (
        SELECT DISTINCT
            subject_name
        FROM act_relationships
        WHERE
            subject_name NOT IN (
                SELECT title
                FROM legislations
            )
    );

select count(*) from act_relationships
WHERE
    subject_name IN (
        SELECT DISTINCT
            subject_name
        FROM act_relationships
        WHERE
            subject_name NOT IN (
                SELECT title
                FROM legislations
            )
    );
