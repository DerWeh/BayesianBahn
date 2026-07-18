# kotlinx.serialization
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.**
-keep,includedescriptorclasses class io.github.derweh.bayesianbahn.**$$serializer { *; }
-keepclassmembers class io.github.derweh.bayesianbahn.** {
    *** Companion;
}
-keepclasseswithmembers class io.github.derweh.bayesianbahn.** {
    kotlinx.serialization.KSerializer serializer(...);
}
