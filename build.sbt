name := "sparkguard"
version := "0.1"
scalaVersion := "2.12.18"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql"   % "3.5.1",
  "org.apache.spark" %% "spark-mllib" % "3.5.1",
  "ml.dmlc" %% "xgboost4j"       % "2.0.3",
  "ml.dmlc" %% "xgboost4j-spark" % "2.0.3"
)

fork := true
javaOptions ++= Seq(
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/java.util=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED"
)
Compile / unmanagedSources / excludeFilter := HiddenFileFilter || "FraudDetectionSVM.scala"
javaOptions += "-Dspark.master=local[*]"
javaOptions += "-Dhadoop.home.dir=C:/hadoop"
javaOptions += "-Djava.library.path=C:/hadoop/bin"
