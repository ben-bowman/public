with inventory as (
    select
        p.productid,
        p.productname,
        p.unitsinstock,
        p.reorderlevel,
        sum(od.quantity) / count(distinct o.orderdate) as avgdailysales
    from {{ source('northwind', 'products') }} p
    join {{ source('northwind', 'order_details') }} od on p.productid = od.productid
    join {{ source('northwind', 'orders') }} o on od.orderid = o.orderid
    group by p.productid, p.productname, p.unitsinstock, p.reorderlevel
)
-- restocking_report
    select
        productid,
        productname,
        unitsinstock,
        reorderlevel,
        avgdailysales,
        case
            when unitsinstock < reorderlevel then reorderlevel - unitsinstock
            else 0
        end as reorder_quantity
    from inventory

