mod support;

use sqlser::Unparser;
use sqlser::dialect::DuckDBDialect;
use support::{ctx, opt_plan};

#[tokio::test]
async fn lowers_a_simple_optimized_plan() {
    let ctx = ctx().await;
    let plan = opt_plan(&ctx, "SELECT c_custkey FROM customer WHERE c_acctbal > 100").await;
    let sql = Unparser::new(&DuckDBDialect::new()).plan_to_sql(&plan).unwrap();
    println!("{sql}");
    assert!(sql.contains("SELECT"));
}
