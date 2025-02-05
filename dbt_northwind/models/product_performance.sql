

SELECT 
    od.ProductID,
    p.ProductName,
    p.CategoryID,
    SUM(od.Quantity) AS total_quantity_sold,
    SUM((od.UnitPrice * od.Quantity * (1-od.Discount)) AS total_revenue,
    p.UnitsInStock,
    p.UnitsOnOrder
FROM {{ source('northwind', 'order_details') }} AS od
JOIN {{ source('northwind', 'products') }} AS p ON od.ProductID = p.ProductID
GROUP BY od.ProductID, p.ProductName, p.CategoryID, p.UnitsInStock, p.UnitsOnOrder

