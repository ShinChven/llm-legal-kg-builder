select count(*) from act_relationships;


select
	act_relationships.subject_name,
	legislations.title,
	legislations.word_count,
	legislations.pdf_link,
	count(*) as relationship_count,
	count(*) * 1.0 / legislations.word_count as relationship_density
from act_relationships
join legislations
	on legislations.title = act_relationships.subject_name
group by act_relationships.subject_name, legislations.title, legislations.pdf_link, legislations.word_count;


-- The mean_relationship_count is the average number of relationships found per legislative act in your dataset.
-- The median_relationship_count represents the "middle" value. If you were to list all the relationship counts for every act in order, the median is the one right in the middle. It's often a more representative measure than the mean when there are some acts with an unusually high number of relationships, as it is not as skewed by these outliers.
WITH relationship_counts AS (
    select
        count(*) as relationship_count
    from act_relationships
    join legislations
        on legislations.title = act_relationships.subject_name
    group by act_relationships.subject_name, legislations.title, legislations.pdf_link, legislations.word_count
)
SELECT
    avg(relationship_count) as mean_relationship_count,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY relationship_count) AS median_relationship_count
FROM relationship_counts;


select * from act_relationships where subject_name = 'Stamp and Cheque Duties Act 1971'



select * from act_relationships where subject_name = 'Maori Purposes Act 1956';
select * from act_relationships where subject_name = 'Employment Relations Act 2000';


select * from act_relationships where object_name ilike '%–%'


select object_name, count(*) from act_relationships GROUP BY object_name ORDER BY count(*) desc;

